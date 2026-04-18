from .path_helper import add_base_to_sys_path
add_base_to_sys_path(2)
import global_config
from .main_func import *
from .model_factory import set_params, get_model
from .memory import AgentMemory
from .router import Router
import json
import os
import pandas as pd
import traceback
from datetime import datetime
from typing import Callable, List
import copy
import dill
from typing import Callable
from IPython.utils import io
import gc

import shutil
import gc



import os
import io
import gc
import dill
import asyncio
import pandas as pd
from typing import Callable, List

async def MALMAS_random_experiments_async(
    task_name: str,
    task: str,
    read_data_class,
    model_name: str,
    metric: str,
    prompt_path_list: list,
    Nround: int,
    min_effective: int=2,
    long_memory_feature_num: int=3,


    temp: float = 1.0,
    other_model:str="",
    verbose: bool = True,
    router_strategy: str = "hybrid",
    router_min_agents: Optional[int] = None,
    router_max_agents: Optional[int] = None,
    router_warmup_rounds: int = 1,
    router_use_llm: bool = False
):
    rows = []
    random_states_list=read_data_class.get_seed_list()
    for random in random_states_list:
        set_params(task=task, model_name=model_name,
                   task_name=task_name, random_state=random, metric=metric,other_model=other_model)

        df_train, df_test, target_column, task_description, description, enrich_description =read_data_class.read_data()

        
        # print(test_Classifier(
        #     df_train, df_test, target_column))
        task_name = global_config.task_name
        model_name = global_config.model_name
        random_state = global_config.data_pre["random_state"]
        cache_dir = f"memory_files/{task+global_config.other_model}/{task_name}/{random_state}"
        os.makedirs(cache_dir, exist_ok=True)
        train_cache_path = f"{cache_dir}/{Nround}_train.csv"
        test_cache_path = f"{cache_dir}/{Nround}_test.csv"

        # 检查缓存是否存在
        if os.path.exists(train_cache_path) and os.path.exists(test_cache_path):
            print("[MALMAS] Loaded cached feature.")
            new_features_train = pd.read_csv(train_cache_path)
            new_features_test = pd.read_csv(test_cache_path)

        else:
            importance_str = get_xgboost_feature_importance(
                df_train, target_column, top_rate=None)
            if verbose:
                train_memory, test_memory = await memory_aware_pipeline_async(
                    df_train=df_train,
                    df_test=df_test,
                    description=description,
                    enrich_description=enrich_description,
                    task_description=task_description,
                    target_column=target_column,
                    dataset_name=task_name,
                    importance_str=importance_str,
                    prompt_path_list=prompt_path_list,
                    gentemp=temp,
                    Nround=Nround,
                    min_effective=min_effective,
                    long_memory_feature_num=long_memory_feature_num,
                    cache_dir=cache_dir,
                    gain_method="gain",
                    iter_or_not=False,
                    router_strategy=router_strategy,
                    router_min_agents=router_min_agents,
                    router_max_agents=router_max_agents,
                    router_warmup_rounds=router_warmup_rounds,
                    router_use_llm=router_use_llm
                )
            else:
                with io.capture_output() as captured:
                    train_memory, test_memory = await memory_aware_pipeline_async(
                        df_train=df_train,
                        df_test=df_test,
                        description=description,
                        enrich_description=enrich_description,
                        task_description=task_description,
                        target_column=target_column,
                        dataset_name=task_name,
                        importance_str=importance_str,
                        prompt_path_list=prompt_path_list,
                        gentemp=temp,
                        Nround=Nround,
                        min_effective=min_effective,
                        long_memory_feature_num=long_memory_feature_num,
                        cache_dir=cache_dir,
                        gain_method="gain",
                        iter_or_not=False,
                        router_strategy=router_strategy,
                        router_min_agents=router_min_agents,
                        router_max_agents=router_max_agents,
                        router_warmup_rounds=router_warmup_rounds,
                        router_use_llm=router_use_llm
                    )

            new_features_train = process_new_features_list(train_memory)
            new_features_test = process_new_features_list(test_memory)
            del train_memory, test_memory

            new_features_train.to_csv(train_cache_path, index=False)
            new_features_test.to_csv(test_cache_path, index=False)

        _, clf_train, clf_test = test_Classifier(
            df_train, df_test, target_column)
        
        _, train_score, test_score = test_Classifier(
            pd.concat([df_train, new_features_train], axis=1),
            pd.concat([df_test, new_features_test], axis=1),
            target_column=target_column
        )

        rows.append({
            f"random_state": random_state,
            f"{model_name}_base": clf_test,
            "MALMAS(ours)": test_score
        })

        del df_train, df_test, task_description, description, enrich_description, new_features_train, new_features_test
        gc.collect()
    df = pd.DataFrame(rows).set_index(f"random_state").reset_index(drop=True).T
    df["mean"] = df.mean(axis=1)
    df["std"] = df.std(axis=1)

    return df


import asyncio
import gc
import dill

async def run_agent_task(
    prompt_path, round_idx, memories, agent_features, description, enrich_description,
    df_train, df_test, task_description, importance_str, gentemp, gain_method,
    target_column, evaluate_func, code_prompt, Nround, min_effective, func_list,agent_gains,global_summary
):
    agent_name = os.path.splitext(os.path.basename(prompt_path))[0]
    memory = memories[agent_name]

    # print(f"\n▶️ Agent: {agent_name} | round: {round_idx+1}")

    with open(prompt_path, "r", encoding="utf-8") as f:
        agent_prompt_template = f.read()

    des = enrich_description if agent_name == "localpattern" else description

    positive, negative = memory.get_positive_negative_columns()

    
    if global_summary is not None:
        agent_prompt_template += f"\nGlobal Conceptual Memory:\n{global_summary}\n"

    prompt_memory_info = ""
    if memory.should_use_memory(round_idx, warmup_round=1):
        prompt_memory_info = memory.generate_prompt_section(
            use_feedback=True,
            use_procedural=False,
        )
        agent_prompt_template += (
            f"\n\nLocal Conceptual Memory for This Agent:\n{memory.conceptual_summary}"
        )

    current_feature_info = f"\n[Valid features generated]{','.join(positive)}\n[Invalid features generated]{','.join(negative)}"
    if round_idx >= (Nround + 1) // 2:
        current_feature_info += "\n\n Ranking of feature importance: \n" + importance_str

    user_prompt = (
        json.dumps(des, ensure_ascii=False, indent=2) +
        "\n\n" + task_description +
        current_feature_info +
        "\n\n" + prompt_memory_info
    )
    while True:
        try:
            response = await asyncio.to_thread(
                generate_response,
                global_config.LLM["llm_model"],
                global_config.LLM["api_key"],
                global_config.LLM["base_url"],
                agent_prompt_template,
                user_prompt,
                gentemp
            )

            if response is None:
                raise Exception("The language model returned an empty response.")
            assistant_content, error_prompt = None, None
            _round = 3
            while _round != 0 :
                
                code = await asyncio.to_thread(
                    generate_response,
                    global_config.LLM["llm_model"],
                    global_config.LLM["api_key"],
                    global_config.LLM["base_url"],
                    code_prompt,
                    response,
                    global_config.LLM["code_temp"],
                    assistant_content, error_prompt
                )
                try:
                    func, _ = extract_and_execute_function(code)
                    new_fea = func(df_train.drop(columns=[target_column]))
                    new_fea_test = func(df_test.drop(columns=[target_column]))
                    break
                except Exception as e:
                    assistant_content = code
                    error_prompt = traceback.format_exc()
                    print(f"[❌code error {agent_name} agent] {str(e)[:50]}")
  
                    _round=_round - 1
                    continue
            if _round==0:
                continue
            response_json = extract_feature_json(response)
            existing = set(positive + negative + df_train.columns.to_list())
            new_columns = [col for col in new_fea.columns if col not in existing]

            if not new_columns:
                print(f"⚠️ [{agent_name}]Skipping. All newly generated features have been evaluated.")
                return 

            new_fea, new_fea_test = new_fea[new_columns], new_fea_test[new_columns]
            func_list.extend([func, new_columns])

            gain = evaluate_func(
                base_features_df=df_train,
                new_features_df=new_fea,
                target_column=target_column,
                verbose=False
            )
            break
        except Exception as e:
            print(f"[❌Val error {agent_name} agent] round: {round_idx+1}:{str(e)[:50]}")
            error_msg = traceback.format_exc()
            
    agent_gains[agent_name] = gain

    for _, row in gain.iterrows():
        fname, each_gain = row["feature"], row[gain_method]
        is_effective = each_gain > 0
        features_des_json = find_feature_metadata(response_json, fname)
        base, ty, transform, logic = [], "", "", None
        if features_des_json:
            base, ty = features_des_json["base_columns"], features_des_json["type"]
            transform, logic = features_des_json["transform"], features_des_json["logic"]

        if is_effective:
            memory.record_procedure(base, transform, fname, ty, logic, round_idx)
            memory.record_feedback(fname, "gain", each_gain, True, round_idx, agent_name, base, ty)
        else:
            memory.record_unused_procedure(base, transform, fname, ty, logic, round_idx)

    current_positive = gain[gain[gain_method] > 0]["feature"].tolist()
    agent_features[agent_name]["train_positive"] = pd.concat(
        [agent_features[agent_name]["train_positive"], new_fea[current_positive]], axis=1)
    agent_features[agent_name]["test_positive"] = pd.concat(
        [agent_features[agent_name]["test_positive"], new_fea_test[current_positive]], axis=1)

    current_negative = gain[gain[gain_method] <= 0]["feature"].tolist()
    agent_features[agent_name]["train_negative"] = pd.concat(
        [agent_features[agent_name]["train_negative"], new_fea[current_negative]], axis=1)
    agent_features[agent_name]["test_negative"] = pd.concat(
        [agent_features[agent_name]["test_negative"], new_fea_test[current_negative]], axis=1)

    memory.generate_conceptual_summary_llm(min_effective)
    


async def memory_aware_pipeline_async(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    description: dict,
    enrich_description: dict,
    task_description: str,
    target_column: str,
    dataset_name: str,
    importance_str: str,
    prompt_path_list: List[str],
    gentemp: float,
    Nround: int,
    min_effective:int,
    long_memory_feature_num: int,
    cache_dir: str,
    gain_method: str = "gain",
    detailfile: str = "_error_file.txt",
    iter_or_not:bool = False,
    router_strategy: str = "hybrid",
    router_min_agents: Optional[int] = None,
    router_max_agents: Optional[int] = None,
    router_warmup_rounds: int = 1,
    router_use_llm: bool = False

):
    df_train = df_train.copy(deep=True)
    df_test = df_test.copy(deep=True)

    
    description = copy.deepcopy(description)
    enrich_description = copy.deepcopy(enrich_description)

    eval_func_map = {
        "regression": evaluate_new_feature_gain_cv,
        "classification": evaluate_new_feature_gain_cv_cls
    }
    try:
        evaluate_func = eval_func_map[global_config.task]
    except KeyError:
        raise ValueError(f"Unsupported task type: {global_config.task}")

    # error = open(dataset_name + detailfile, "w", encoding="utf-8")
    func_list=[]
    with open("prompt_files/codegeneration.txt", "r", encoding="utf-8") as f:
        code_prompt = f.read()


    # Initialize Router Agent
    router = Router(
        prompt_path_list=prompt_path_list,
        strategy=router_strategy,
        min_agents=router_min_agents,
        max_agents=router_max_agents,
        warmup_rounds=router_warmup_rounds,
        use_llm=router_use_llm
    )

    # Initialize all agent memories (even if not selected, they may be selected later)
    memories = {}
    agent_features = {}
    if iter_or_not:
        train_iter_features=[]
        test_iter_features=[]

    for prompt_path in prompt_path_list:
        agent_name = os.path.splitext(os.path.basename(prompt_path))[0]
        memories[agent_name] = AgentMemory(agent_name, dataset_name, Nround=Nround,cache_dir=cache_dir)
        agent_features[agent_name] = {
            "train_positive": pd.DataFrame(),
            "test_positive": pd.DataFrame(),
            "train_negative": pd.DataFrame(),
            "test_negative": pd.DataFrame()
        }
    print(f"\n================ Initialized AgentMemory ================\n")

    global_summary = None
    for round_idx in range(Nround):
        print(f"\n🔁  ===== The {round_idx + 1}-th round =====")
        agent_gains={}

        # Router selects active agent subset for this round
        selected_prompt_paths = router.select_agents(
            round_idx=round_idx,
            df=df_train if round_idx == router_warmup_rounds else None,
            description=description,
            enrich_description=enrich_description,
            task_description=task_description
        )

        tasks = [
            run_agent_task(prompt_path, round_idx, memories, agent_features, description, enrich_description,
                           df_train, df_test, task_description, importance_str, gentemp, gain_method,
                           target_column, evaluate_func, code_prompt, Nround, min_effective, func_list,agent_gains,global_summary)
            for prompt_path in selected_prompt_paths
        ]
        await asyncio.gather(*tasks)

        # Update router with agent performance from this round
        for agent_name, gain_df in agent_gains.items():
            if not gain_df.empty and gain_method in gain_df.columns:
                avg_gain = gain_df[gain_method].mean()
                router.update_performance(agent_name, avg_gain)

        global_summary = AgentMemory.generate_global_conceptual_summary(memories, task_description)
        df_train, df_test, description, enrich_description = persist_top_features_and_update_description(
            df_train, df_test, agent_gains, agent_features, description, enrich_description,
            target_column, long_memory_feature_num
        )
        
    # 保存记忆和特征列表
    for memory in memories.values():
        memory.save_memory()

    with open(f'{cache_dir}/func_list.pkl', 'wb') as f:
        dill.dump(func_list, f)

    # Save router summary
    router_summary = router.get_summary()
    with open(f'{cache_dir}/router_summary.json', 'w', encoding='utf-8') as f:
        json.dump(router_summary, f, indent=2, ensure_ascii=False)
    print(f"\n[Router] Summary: {router_summary['selection_counts']}")

    agent_feature_train_list = [agent_features[a]["train_positive"] for a in memories.keys()]
    agent_feature_test_list = [agent_features[a]["test_positive"] for a in memories.keys()]

    gc.collect()
    return agent_feature_train_list, agent_feature_test_list




