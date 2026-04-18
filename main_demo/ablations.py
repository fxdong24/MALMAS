from .path_helper import add_base_to_sys_path
add_base_to_sys_path(2)
import global_config
from .main_func import *
from .model_factory import set_params, get_model
from .memory import AgentMemory
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
from pandas.errors import EmptyDataError
import shutil
import gc

import os
import io
import gc
import dill
import asyncio
import pandas as pd
from typing import Callable, List

async def MALMAS_random_experiments_without_memory_async(
    task_name: str,
    task: str,
    read_data_class,
    model_name: str,
    metric: str,
    prompt_path_list: list,
    Nround: int,
    temp: float = 1.0,
    verbose: bool = True,
    placeholder="MALMAS_without_memory"
):
    rows = []
    random_states_list=read_data_class.get_seed_list()
    for random in random_states_list:
        set_params(task=task, model_name=model_name,
                   task_name=task_name, random_state=random, metric=metric,)

        df_train, df_test, target_column, task_description, description, enrich_description = read_data_class.read_data()

        
        task_name = global_config.task_name
        model_name = global_config.model_name
        random_state = global_config.data_pre["random_state"]
        cache_dir = f"memory_files/ablation/{task_name}/{placeholder}/{random_state}"
        os.makedirs(cache_dir, exist_ok=True)
        train_cache_path = f"{cache_dir}/{Nround}_train.csv"
        test_cache_path = f"{cache_dir}/{Nround}_test.csv"

        # 检查缓存是否存在
        if os.path.exists(train_cache_path) and os.path.exists(test_cache_path):
            print("[MALMAS Ablation] Loaded cached feature.")
            try:
                new_features_train = pd.read_csv(train_cache_path)
                new_features_test = pd.read_csv(test_cache_path)
            except EmptyDataError:
                new_features_train=pd.DataFrame()
                new_features_test=pd.DataFrame()

        else:
            importance_str = get_xgboost_feature_importance(
                df_train, target_column, top_rate=None)
            if verbose:
                train_memory, test_memory = await pipeline_async(
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
                    cache_dir=cache_dir,
                    gain_method="gain",
                    iter_or_not=False
                )
            else:
                with io.capture_output() as captured:
                    train_memory, test_memory = await pipeline_async(
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
                        cache_dir=cache_dir,
                        gain_method="gain",
                        iter_or_not=False
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
    prompt_path, round_idx, agent_features, description, enrich_description,
    df_train, df_test, task_description, importance_str, gentemp, gain_method,
    target_column, evaluate_func, code_prompt, Nround,  func_list,agent_gains
):
    agent_name = os.path.splitext(os.path.basename(prompt_path))[0]


    with open(prompt_path, "r", encoding="utf-8") as f:
        agent_prompt_template = f.read()

    des = enrich_description if agent_name == "localpattern" else description


    current_feature_info=""
    if round_idx >= (Nround + 1) // 2:
        current_feature_info += "\n\n Ranking of feature importance: \n" + importance_str

    user_prompt = (
        json.dumps(des, ensure_ascii=False, indent=2) +
        "\n\n" + task_description
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

            func_list.extend([func, new_fea.columns])

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




async def pipeline_async(
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

    cache_dir: str,
    gain_method: str = "gain",
    detailfile: str = "_error_file.txt",
    iter_or_not:bool = False

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

    agent_features = {}
    if iter_or_not:
        train_iter_features=[]
        test_iter_features=[]
    name_list=[]
    for prompt_path in prompt_path_list:
        agent_name = os.path.splitext(os.path.basename(prompt_path))[0]
        name_list.append(agent_name)
        agent_features[agent_name] = {
            "train_positive": pd.DataFrame(),
            "test_positive": pd.DataFrame(),
            "train_negative": pd.DataFrame(),
            "test_negative": pd.DataFrame()
        }

    for round_idx in range(Nround):
        print(f"\n🔁  ===== The {round_idx + 1}-th round =====")
        agent_gains={}
        tasks = [
            run_agent_task(prompt_path, round_idx,  agent_features, description, enrich_description,
                           df_train, df_test, task_description, importance_str, gentemp, gain_method,
                           target_column, evaluate_func, code_prompt, Nround,func_list,agent_gains)
            for prompt_path in prompt_path_list
        ]
        await asyncio.gather(*tasks)
        

    with open(f'{cache_dir}/func_list.pkl', 'wb') as f:
        dill.dump(func_list, f)

    agent_feature_train_list = [agent_features[a]["train_positive"] for a in name_list]
    agent_feature_test_list = [agent_features[a]["test_positive"] for a in name_list]
    del agent_features
    gc.collect()
    return agent_feature_train_list, agent_feature_test_list


