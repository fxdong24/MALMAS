from .path_helper import add_base_to_sys_path

add_base_to_sys_path(2)
import os
import pandas as pd
import numpy as np
from .LLMFE_demo.LLMFE import run_llmfe_feature_engineering
from main_demo.model_factory import set_params, get_model
from main_demo.main_func import test_Classifier,test_Regressor
import global_config
from autofeat import AutoFeatRegressor, AutoFeatClassifier
from sklearn.metrics import mean_squared_error, roc_auc_score,accuracy_score
from main_demo.main_func import preprocess_X
from xgboost import XGBRegressor, XGBClassifier
import sys
from caafe import CAAFEClassifier
from sklearn.model_selection import train_test_split
from openfe import OpenFE
import featuretools as ft
from .utils_xg import evaluate, gen_prompt, tree_to_code, get_cart, add_column, load_model, use_api,filter_and_normalize_proba
import traceback
from IPython.utils import io
from pandas.errors import EmptyDataError
import gc
import copy
import re
import random
def run_baseline_experiments_for_Regressor(
        task,
        model_name,
        task_name,
        read_data_class,
        metric="nrmse",
        verbose=True
):
    def run_baseline_func(
            task,
            model_name,
            task_name,
            read_data_class,
            metric
    ):
        random_states_list=read_data_class.get_seed_list()
        rows = []
        for random_state in random_states_list:
            set_params(task=task, model_name=model_name,
                       task_name=task_name, random_state=random_state,metric=metric,)

            df_train, df_test, target_column, task_description, description, enrich_description = read_data_class.read_data()

            _, clf_train, clf_test = test_Regressor(
                df_train, df_test, target_column)
            error_msg = None
            try:
                dfs_train, dfs_test = generate_dfs_features_and_evaluate(
                    df_train, df_test, target_column)
            except EmptyDataError:
                dfs_train, dfs_test = clf_train, clf_test
            # except Exception:
            #     dfs_train, dfs_test = None, None
            #     error_msg = traceback.format_exc()
            try:
                autofeat_train, autofeat_test = run_autofeat_with_preprocessing(
                    df_train, df_test, target_column, feateng_steps=1, get_model_or=True)
            except EmptyDataError:
                autofeat_train, autofeat_test = clf_train, clf_test
            # except Exception:
            #     autofeat_train, autofeat_test = None, None
            #     error_msg = traceback.format_exc()
            
            try:
                openfe_train, openfe_test = run_openfe_pipeline(
                    df_train, df_test, target_column)
            except EmptyDataError:
                openfe_train, openfe_test = clf_train, clf_test
            except Exception:
                openfe_train, openfe_test = None, None
                error_msg = traceback.format_exc()
            
            try:
                
                llmfe_train, llmfe_test = run_or_load_llmfe_pipeline(
                    df_train,
                    df_test,
                    target_column,
                    description,
                    task_description,
                    task=task,
                    max_sample_num = 20
                )
            except EmptyDataError:
                llmfe_train, llmfe_test = clf_train, clf_test
            # except Exception:
            #     llmfe_train, llmfe_test = None, None
            #     error_msg = traceback.format_exc()

            if error_msg is not None:
                print(error_msg)
                sys.exit(1)

            rows.append({
                f"seed_{metric}": random_state,
                f"{model_name}_base": clf_test,
                "DFS": dfs_test,
                "AutoFeat": autofeat_test,
                "OpenFE": openfe_test,
                "LLMFE": llmfe_test
            })
        return rows

    if verbose:
        rows = run_baseline_func(
                                 task,
                                 model_name,
                                 task_name,
                                 read_data_class, 
                                metric)
    else:
        with io.capture_output() as captured:
            rows = run_baseline_func(
                                     task,
                                     model_name,
                                     task_name,
                                     read_data_class,metric )
    
    df = pd.DataFrame(rows).set_index(f"seed_{metric}").reset_index(drop=True).T
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    # df[numeric_cols] = df[numeric_cols] * 100

    # 添加均值和方差列
    df["mean"] = df.mean(axis=1)
    df["std"] = df.std(axis=1)

    return df
def run_baseline_experiments_for_other_model(
        task,
        model_name,
        task_name,
        read_data_class,
        other_model,
        metric="auc",
        verbose=True,
):
    def run_baseline_func(
            task,
            model_name,
            task_name,
            read_data_class,
            metric
    ):
        random_states_list=read_data_class.get_seed_list()
        rows = []
        for random_state in random_states_list:
            set_params(task=task, model_name=model_name,
                       task_name=task_name, random_state=random_state,metric=metric,other_model=other_model)

            df_train, df_test, target_column, task_description, description, enrich_description = read_data_class.read_data()

            _, clf_train, clf_test = test_Classifier(
                df_train, df_test, target_column)
            error_msg = None

            try:
                dfs_train, dfs_test = generate_dfs_features_and_evaluate(
                    df_train, df_test, target_column)
            except EmptyDataError:
                dfs_train, dfs_test = clf_train, clf_test
            # except Exception:
            #     dfs_train, dfs_test = None, None
            #     error_msg = traceback.format_exc()

            try:
                autofeat_train, autofeat_test = run_autofeat_with_preprocessing(
                    df_train, df_test, target_column, feateng_steps=1, get_model_or=True)
            except EmptyDataError:
                autofeat_train, autofeat_test = clf_train, clf_test
            # except Exception:
            #     autofeat_train, autofeat_test = None, None
            #     error_msg = traceback.format_exc()

            try:
                openfe_train, openfe_test = run_openfe_pipeline(
                    df_train, df_test, target_column)
            except EmptyDataError:
                openfe_train, openfe_test = clf_train, clf_test
            # except Exception:
            #     openfe_train, openfe_test = None, None
            #     error_msg = traceback.format_exc()
            try:
                caafe_train, caafe_test = run_caafe_classifier(
                    df_train, df_test, target_column, task_description)
            except EmptyDataError:
                caafe_train, caafe_test = clf_train, clf_test
            # except Exception:
            #     caafe_train, caafe_test = None, None
            #     error_msg = traceback.format_exc()

            try:
                octree_train, octree_test = run_octree_auc_pipeline(
                    df_train, df_test, target_column, 20)
            except EmptyDataError:
                octree_train, octree_test = clf_train, clf_test
            # except Exception:
            #     octree_train, octree_test = None, None
            #     error_msg = traceback.format_exc()

            try:
                llmfe_train, llmfe_test = run_or_load_llmfe_pipeline(
                    df_train,
                    df_test,
                    target_column,
                    description,
                    task_description,
                    task=task,
                    max_sample_num = 20
                )
            except EmptyDataError:
                llmfe_train, llmfe_test = clf_train, clf_test
            # except Exception:
            #     llmfe_train, llmfe_test = None, None
            #     error_msg = traceback.format_exc()

            if error_msg is not None:
                print(error_msg)
                sys.exit(1)
            rows.append({
                f"seed_{metric}": random_state,
                f"{model_name}_base": clf_test,
                "DFS": dfs_test,
                "AutoFeat": autofeat_test,
                "OpenFE": openfe_test,
                "CAAFE": caafe_test,
                "OCTree": octree_test,
                "LLMFE": llmfe_test
            })
        return rows

    if verbose:
        rows = run_baseline_func(
                                 task,
                                 model_name,
                                 task_name,
                                 read_data_class, 
                                metric)
    else:
        with io.capture_output() as captured:
            rows = run_baseline_func(
                                     task,
                                     model_name,
                                     task_name,
                                     read_data_class,metric )
    
    df = pd.DataFrame(rows).set_index(f"seed_{metric}").reset_index(drop=True).T
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

    # 添加均值和方差列
    df["mean"] = df.mean(axis=1)
    df["std"] = df.std(axis=1)

    return df


def run_baseline_experiments(
        task,
        model_name,
        task_name,
        read_data_class,
        metric="auc",
        verbose=True
):
    def run_baseline_func(
            task,
            model_name,
            task_name,
            read_data_class,
            metric
    ):
        random_states_list=read_data_class.get_seed_list()
        rows = []
        for random_state in random_states_list:
            set_params(task=task, model_name=model_name,
                       task_name=task_name, random_state=random_state,metric=metric,)

            df_train, df_test, target_column, task_description, description, enrich_description = read_data_class.read_data()

            _, clf_train, clf_test = test_Classifier(
                df_train, df_test, target_column)
            error_msg = None
            try:
                dfs_train, dfs_test = generate_dfs_features_and_evaluate(
                    df_train, df_test, target_column)
            except EmptyDataError:
                dfs_train, dfs_test = clf_train, clf_test
            # except Exception:
            #     dfs_train, dfs_test = None, None
            #     error_msg = traceback.format_exc()

            try:
                autofeat_train, autofeat_test = run_autofeat_with_preprocessing(
                    df_train, df_test, target_column, feateng_steps=1, get_model_or=True)
            except EmptyDataError:
                autofeat_train, autofeat_test = clf_train, clf_test
            # except Exception:
            #     autofeat_train, autofeat_test = None, None
            #     error_msg = traceback.format_exc()

            try:
                openfe_train, openfe_test = run_openfe_pipeline(
                    df_train, df_test, target_column)
            except EmptyDataError:
                openfe_train, openfe_test = clf_train, clf_test
            # except Exception:
            #     openfe_train, openfe_test = None, None
            #     error_msg = traceback.format_exc()

            try:
                caafe_train, caafe_test = run_caafe_classifier(
                    df_train, df_test, target_column, task_description)
            except EmptyDataError:
                caafe_train, caafe_test = clf_train, clf_test
            # except Exception:
            #     caafe_train, caafe_test = None, None
            #     error_msg = traceback.format_exc()

            try:
                octree_train, octree_test = run_octree_auc_pipeline(
                    df_train, df_test, target_column, 20)
            except EmptyDataError:
                octree_train, octree_test = clf_train, clf_test
            # except Exception:
            #     octree_train, octree_test = None, None
            #     error_msg = traceback.format_exc()

            try:
                llmfe_train, llmfe_test = run_or_load_llmfe_pipeline(
                    df_train,
                    df_test,
                    target_column,
                    description,
                    task_description,
                    task=task,
                    max_sample_num = 20
                )
            except EmptyDataError:
                llmfe_train, llmfe_test = clf_train, clf_test
            # except Exception:
            #     llmfe_train, llmfe_test = None, None
            #     error_msg = traceback.format_exc()

            if error_msg is not None:
                print(error_msg)
                sys.exit(1)

            rows.append({
                f"seed_{metric}": random_state,
                f"{model_name}_base": clf_test,
                "DFS": dfs_test,
                "AutoFeat": autofeat_test,
                "OpenFE": openfe_test,
                "CAAFE": caafe_test,
                "OCTree": octree_test,
                "LLMFE": llmfe_test
            })
        return rows

    if verbose:
        rows = run_baseline_func(
                                 task,
                                 model_name,
                                 task_name,
                                 read_data_class, 
                                metric)
    else:
        with io.capture_output() as captured:
            rows = run_baseline_func(
                                     task,
                                     model_name,
                                     task_name,
                                     read_data_class,metric )
    
    df = pd.DataFrame(rows).set_index(f"seed_{metric}").reset_index(drop=True).T
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df["mean"] = df.mean(axis=1)
    df["std"] = df.std(axis=1)

    return df
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error

def evaluate_model(train_X, train_y, test_X, test_y):
    """
    训练模型并计算 train/test 的评分，根据 global_config.task 和 global_config.metric 自动选择指标
    """

    def clean_feature_names(df):
        cleaned_columns = []
        for col in df.columns:
            # Convert to string if not already
            col_str = str(col) if not isinstance(col, str) else col
            cleaned_columns.append(re.sub(r'[^a-zA-Z0-9_]', '_', col_str))
        df.columns = cleaned_columns
        return df
    train_X = clean_feature_names(train_X)
    test_X = clean_feature_names(test_X)

    model = get_model()
    model.fit(train_X, train_y)

    if global_config.task == 'classification':
        num_classes = len(np.unique(train_y))

        if global_config.metric == 'acc':
            # 用 predict
            train_pred = model.predict(train_X)
            test_pred = model.predict(test_X)

            train_score = accuracy_score(train_y, train_pred)
            test_score = accuracy_score(test_y, test_pred)

        elif global_config.metric == 'auc':
            # 用 predict_proba
            y_train_proba = model.predict_proba(train_X)
            y_test_proba = model.predict_proba(test_X)

            if num_classes == 2:
                train_score = roc_auc_score(train_y, y_train_proba[:, 1])
                test_score = roc_auc_score(test_y, y_test_proba[:, 1])
            else:
                train_score = roc_auc_score(train_y, y_train_proba, multi_class='ovr', average='macro')
                test_score = roc_auc_score(test_y, y_test_proba, multi_class='ovr', average='macro')

        else:
            raise ValueError(f"Unsupported classification metric: {global_config.metric}")

    elif global_config.task == 'regression':
        y_train_pred = model.predict(train_X)
        y_test_pred = model.predict(test_X)

        train_score = nrmse(train_y, y_train_pred)
        test_score = nrmse(test_y, y_test_pred)

    else:
        raise ValueError(f"Unsupported task type: {global_config.task}")
    del train_X, train_y, test_X, test_y
    gc.collect()
    return train_score, test_score

def nrmse(y_true,y_pred):
    rmse=np.sqrt(mean_squared_error(y_true, y_pred))
    norm=np.mean(y_true)
    return rmse/norm

def remove_duplicate_columns(df):
    # 找出重复的列
    duplicate_cols = df.columns[df.columns.duplicated()]
    
    if len(duplicate_cols) > 0:
        df = df.loc[:, ~df.columns.duplicated(keep='first')]
    return df
def run_or_load_llmfe_pipeline(
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        target_column: str,
        description: dict,
        task_description: str,
        task: str,
        max_sample_num: int = 20

):
    task_name = global_config.task_name
    task=global_config.task
    model_name = global_config.model_name
    seed = global_config.data_pre["random_state"]
    save_dir = f"baselines/{task+global_config.other_model}/{task_name}"
    os.makedirs(save_dir, exist_ok=True)

    new_train_path = os.path.join(save_dir, f"LLMFE__new_train_{seed}.csv")
    new_test_path = os.path.join(save_dir, f"LLMFE__new_test_{seed}.csv")
    final_code_path = os.path.join(save_dir, f"LLMFE_final_modify_features_{seed}.py")

    # 1️⃣ 加载或生成新特征
    if os.path.exists(new_train_path) and os.path.exists(new_test_path):
        print("[LLMFE] Loaded cached feature.")
        X_train_new = pd.read_csv(new_train_path)
        X_test_new = pd.read_csv(new_test_path)

    else:
        
        df_train=remove_duplicate_columns(df_train)
        df_test=remove_duplicate_columns(df_test)
        
        print("[LLMFE] runing function to get the \"modify_features.py\" file.")
        run_llmfe_feature_engineering(
            df_train=df_train,
            target_column=target_column,
            task_name=task_name,
            description=description,
            task_description=task_description,
            task=task,
            seed=seed,
            file_model_name=model_name,
            use_api=True,
            api_model=global_config.LLM["llm_model"],
            api_key=global_config.LLM["api_key"],
            other_model=global_config.other_model,
            max_sample_num=max_sample_num,
        )

        local_env = {}
        with open(final_code_path, "r") as f:
            code_str = f.read()
        exec(code_str, globals(), local_env)

        modify_features = local_env["modify_features"]
        
        df_train, df_test, _ = preprocess_X(df_train, df_test)
        
        X_train_full = modify_features(df_train.drop(columns=[target_column]))
        X_test_full = modify_features(df_test.drop(columns=[target_column]))


        original_features = df_train.drop(columns=[target_column]).columns.tolist()
        new_features = [col for col in X_train_full.columns if col not in original_features]

        X_train_new = X_train_full[new_features]
        X_test_new = X_test_full[new_features]

        X_train_new.to_csv(new_train_path, index=False)
        X_test_new.to_csv(new_test_path, index=False)


    X_train_combined = pd.concat([df_train, X_train_new], axis=1)
    X_test_combined = pd.concat([df_test, X_test_new], axis=1)
    X_train_combined=remove_duplicate_columns(X_train_combined)
    X_test_combined=remove_duplicate_columns(X_test_combined)
    X_train_combined, X_test_combined, _ = preprocess_X(X_train_combined, X_test_combined)
    

    y_train = X_train_combined[target_column].values
    y_test = X_test_combined[target_column].values
    
    X_train_combined=X_train_combined.drop(columns=[target_column])
    X_test_combined=X_test_combined.drop(columns=[target_column])
    # 3️⃣ AUC 验证


    return evaluate_model(X_train_combined,y_train,X_test_combined,y_test)


def run_caafe_classifier(
        df_train,
        df_test,
        target_column,
        task_description,
        model=None,
        iterations=10,
):
    # 设置路径
    task_name = global_config.task_name
    task=global_config.task
    model_name = global_config.model_name
    random_state = global_config.data_pre["random_state"]
    cache_dir = f"baselines/{task+global_config.other_model}/{task_name}"
    os.makedirs(cache_dir, exist_ok=True)
    # print("===============================caafe运行了================================")
    df_train, df_test, _ = preprocess_X(df_train, df_test)
    train_cache_path = f"{cache_dir}/caafe_train_{random_state}.csv"
    test_cache_path = f"{cache_dir}/caafe_test_{random_state}.csv"

    if model is None:
        model = get_model()

    # 检查缓存是否存在
    if os.path.exists(train_cache_path) and os.path.exists(test_cache_path):
        print("[CAAFE] Loaded cached feature.")

        df_train_full = pd.read_csv(train_cache_path)
        df_test_full = pd.read_csv(test_cache_path)

    else:
        caafe_clf = CAAFEClassifier(
            base_classifier=model,
            optimization_metric="auc",
            llm_model=global_config.LLM['llm_model'],
            iterations=iterations
        )

        caafe_clf.fit_pandas(
            df_train,
            target_column_name=target_column,
            dataset_description=task_description
        )
        # 使用 CAAFE 的特征生成方法
        df_train_full = pd.concat([pd.DataFrame(caafe_clf.predict_preprocess(df_train.drop(columns=[target_column]))),
                                   df_train[target_column]], axis=1)
        df_test_full = pd.concat(
            [pd.DataFrame(caafe_clf.predict_preprocess(df_test.drop(columns=[target_column]))), df_test[target_column]],
            axis=1)

        df_train_full.to_csv(train_cache_path, index=False)
        df_test_full.to_csv(test_cache_path, index=False)
        print("[CAAFE] Features generated and saved to cache.")
    df_train_full, df_test_full, _ = preprocess_X(df_train_full, df_test_full)

    # 拆分特征与标签
    X_train = df_train_full.drop(columns=[target_column])
    y_train = df_train_full[target_column]
    X_test = df_test_full.drop(columns=[target_column])
    y_test = df_test_full[target_column]

    return evaluate_model(X_train,y_train,X_test,y_test)



def run_autofeat_with_preprocessing(
        df_train,
        df_test,
        target_column,
        feateng_steps=1,
        get_model_or=False
):


    # ========== 准备路径 ==========
    task_name = global_config.task_name
    task=global_config.task
    random_state = global_config.data_pre["random_state"]
    task_type = global_config.task
    model_name = global_config.model_name

    cache_dir = os.path.join("baselines", task, task_name)
    os.makedirs(cache_dir, exist_ok=True)
    train_feat_path = os.path.join(cache_dir, f"autofeat_train_{random_state}.csv")
    test_feat_path = os.path.join(cache_dir, f"autofeat_test_{random_state}.csv")
    
    df_train, df_test, _ = preprocess_X(df_train, df_test)
    # ========== 原始数据处理 ==========
    X_train_raw = df_train.drop(columns=[target_column])
    X_test_raw = df_test.drop(columns=[target_column])
    y_train = df_train[target_column]
    y_test = df_test[target_column]
    

    # ========== 是否读取缓存 ==========
    if os.path.exists(train_feat_path) and os.path.exists(test_feat_path):
        print("[AutoFeat] Loaded cached features.")
        X_train_feat = pd.read_csv(train_feat_path)
        X_test_feat = pd.read_csv(test_feat_path)
    else:
        # ========== 特征构造 ==========
        if task_type == "regression":
            model = AutoFeatRegressor(feateng_steps=feateng_steps, verbose=1,)
        else:
            model = AutoFeatClassifier(feateng_steps=feateng_steps, verbose=1,)
        from sklearn.linear_model import LogisticRegression

        X_train_raw = X_train_raw.replace([np.inf, -np.inf], 0).fillna(0)

        model.fit(X_train_raw, y_train)          # 第一步：仅拟合模型
        X_train_trans = model.transform(X_train_raw)  # 第二步：转换训练集
        X_test_trans = model.transform(X_test_raw)    # 第三步：转换测试集
        # X_train_trans = model.fit_transform(X_train_raw, y_train)
        # X_test_trans = model.transform(X_test_raw)

        # 仅保留新特征部分
        new_feat_names = [col for col in X_train_trans.columns if col not in X_train_raw.columns]
        X_train_feat = X_train_trans[new_feat_names].copy()
        X_test_feat = X_test_trans[new_feat_names].copy()

        X_train_feat.to_csv(train_feat_path, index=False)
        X_test_feat.to_csv(test_feat_path, index=False)
        print("[AutoFeat] Features generated and cached.")

    # ========== 特征合并 ==========
    X_train_full = pd.concat([X_train_raw.reset_index(drop=True), X_train_feat], axis=1)
    X_test_full = pd.concat([X_test_raw.reset_index(drop=True), X_test_feat], axis=1)
    X_train_full, X_test_full, _ = preprocess_X(X_train_full, X_test_full)
    
    return evaluate_model(X_train_full,y_train,X_test_full,y_test)



def run_openfe_pipeline(df_train, df_test, target_column):
    """
    使用 OpenFE 自动特征工程，并在给定训练/测试集上评估性能（含缓存机制）

    参数：
    - df_train, df_test: 带标签的训练测试集 DataFrame，不会被修改
    - target_column: 目标变量列名

    返回：
    - train_score, test_score: 回归任务为 RMSE，分类任务为 AUC
    """
    import re

    # 保留字母、数字和下划线
    df_train.columns = [re.sub(r'[^a-zA-Z0-9_]', '', col) for col in df_train.columns]
    df_test.columns = [re.sub(r'[^a-zA-Z0-9_]', '', col) for col in df_test.columns]

    def ensure_valid_dtypes(df):
        for c in df.columns:
            if df[c].dtype == 'object' or str(df[c].dtype) == 'category':
                try:
                    df[c] = df[c].astype(float)
                except:
                    df[c] = df[c].astype(str).astype('category').cat.codes
        return df

    task = global_config.task
    
    task_name = global_config.task_name
    random_state = global_config.data_pre["random_state"]
    model_name = global_config.model_name
    cache_dir = os.path.join("baselines", task, task_name)
    os.makedirs(cache_dir, exist_ok=True)

    train_cache = os.path.join(cache_dir, f"openfe_train_{random_state}.csv")
    test_cache = os.path.join(cache_dir, f"openfe_test_{random_state}.csv")
    
    
    df_train, df_test, _ = preprocess_X(df_train, df_test)
    X_train_raw = df_train.drop(columns=[target_column])
    X_test_raw = df_test.drop(columns=[target_column])
    y_train = df_train[target_column]
    y_test = df_test[target_column]

    if os.path.exists(train_cache) and os.path.exists(test_cache):
        print("[OpenFE] Loaded cached features.")
        X_train_feat = pd.read_csv(train_cache)
        X_test_feat = pd.read_csv(test_cache)
    else:
        temp_cache_path=f"openfe_tmp_{target_column}.feather"
        try:
            
            fe = OpenFE()
            fe.fit(
                data=X_train_raw,
                label=y_train,
                task=task,
                n_jobs=4,
                verbose=True,
                tmp_save_path=temp_cache_path
            )

            X_train_all, X_test_all = fe.transform(
                X_train=X_train_raw,
                X_test=X_test_raw,
                new_features_list=fe.new_features_list,
                n_jobs=4
            )
        finally:
            if os.path.exists(temp_cache_path):
                os.remove(temp_cache_path)
            
        new_cols = [col for col in X_train_all.columns if col not in X_train_raw.columns]
        X_train_feat = X_train_all[new_cols].copy()
        X_test_feat = X_test_all[new_cols].copy()

        X_train_feat = ensure_valid_dtypes(X_train_feat)
        X_test_feat = ensure_valid_dtypes(X_test_feat)

        # 保存新特征
        X_train_feat.to_csv(train_cache, index=False)
        X_test_feat.to_csv(test_cache, index=False)
        print("[OpenFE] Features generated and cached.")

    X_train_full = pd.concat([X_train_raw.reset_index(drop=True), X_train_feat], axis=1)
    X_test_full = pd.concat([X_test_raw.reset_index(drop=True), X_test_feat], axis=1)

    # 再次统一格式（确保类型、编码一致）
    X_train_full, X_test_full, _ = preprocess_X(X_train_full, X_test_full)
    return evaluate_model(X_train_full,y_train,X_test_full,y_test)
    


def generate_dfs_features_and_evaluate(
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        target_column: str,
        max_depth: int = 2,
        drop_cols: list = None,
        verbose: bool = True
):
    task = global_config.task
    drop_cols = drop_cols or []

    task_name = global_config.task_name
    task=global_config.task
    random_state = global_config.data_pre["random_state"]
    model_name = global_config.model_name
    cache_dir = os.path.join("baselines", task, task_name)
    os.makedirs(cache_dir, exist_ok=True)

    train_cache = os.path.join(cache_dir, f"dfs_train_{random_state}.csv")
    test_cache = os.path.join(cache_dir, f"dfs_test_{random_state}.csv")
    df_train, df_test, _ = preprocess_X(df_train, df_test)
    # 提取目标列
    y_train = df_train[target_column]
    y_test = df_test[target_column]
    df_train_raw = df_train.drop(columns=[target_column])
    df_test_raw = df_test.drop(columns=[target_column])

    if os.path.exists(train_cache) and os.path.exists(test_cache):
        if verbose:
            print("[DFS] Loaded cached features.")
        X_train_feat = pd.read_csv(train_cache)
        X_test_feat = pd.read_csv(test_cache)
    else:
        df_train_raw = df_train_raw.copy()
        df_test_raw = df_test_raw.copy()
        df_train_raw["index"] = df_train_raw.index
        df_test_raw["index"] = df_test_raw.index

        es_train = ft.EntitySet(id="dfs_train")
        es_train = es_train.add_dataframe(
            dataframe_name="base",
            dataframe=df_train_raw,
            index="index"
        )

        es_test = ft.EntitySet(id="dfs_test")
        es_test = es_test.add_dataframe(
            dataframe_name="base",
            dataframe=df_test_raw,
            index="index"
        )

        trans_primitives = [
            'add_numeric',
            'subtract_numeric'
        ]
        agg_primitives = [
            # 'sum',
            'mean',
            'std'
        ]

        X_train_feat, _ = ft.dfs(
            entityset=es_train,
            target_dataframe_name="base",
            trans_primitives=trans_primitives,
            agg_primitives=agg_primitives,
            max_depth=max_depth,
            verbose=False
        )

        X_test_feat, _ = ft.dfs(
            entityset=es_test,
            target_dataframe_name="base",
            trans_primitives=trans_primitives,
            agg_primitives=agg_primitives,
            max_depth=max_depth,
            verbose=False
        )

        X_train_feat.to_csv(train_cache, index=False)
        X_test_feat.to_csv(test_cache, index=False)
        if verbose:
            print("[DFS] Features generated and cached.")

    X_train_trans, X_test_trans, _ = preprocess_X(X_train_feat, X_test_feat, drop_cols=drop_cols)
    
    X_train_trans = X_train_trans.replace([np.inf, -np.inf], np.nan)
    X_test_trans = X_test_trans.replace([np.inf, -np.inf], np.nan)
    # print(X_train_trans.shape,y_train.shape,X_test_trans.shape,y_test.shape)
    return evaluate_model(X_train_trans,y_train,X_test_trans,y_test)


from sklearn.preprocessing import LabelEncoder


def preprocess_X_for_octree(X_train, X_test=None, drop_cols=None):
    """
    预处理训练集和测试集特征，保证训练集fit的编码器能应用于测试集。
    使用均值填充数值型缺失值。
    """
    X_train = X_train.copy()
    drop_cols = drop_cols or []

    # 删除指定列
    X_train = X_train.drop(columns=drop_cols, errors='ignore')
    if X_test is not None:
        X_test = X_test.copy()
        X_test = X_test.drop(columns=drop_cols, errors='ignore')

    # 填充训练集 - 类别型
    for col in X_train.select_dtypes(include=['object', 'category']).columns:
        if pd.api.types.is_categorical_dtype(X_train[col]):
            if 'NA' not in X_train[col].cat.categories:
                X_train[col] = X_train[col].cat.add_categories(['NA'])
        X_train[col] = X_train[col].fillna('NA')

    # 填充训练集 - 数值型（使用均值）
    num_means = {}
    for col in X_train.select_dtypes(include='number').columns:
        X_train[col] = X_train[col].replace([np.inf, -np.inf], np.nan)
        mean_value = X_train[col].mean()
        num_means[col] = mean_value
        X_train[col] = X_train[col].fillna(mean_value)

    # 填充测试集
    if X_test is not None:
        for col in X_test.select_dtypes(include=['object', 'category']).columns:
            if col in X_train.columns:
                if pd.api.types.is_categorical_dtype(X_train[col]):
                    X_test[col] = X_test[col].astype(pd.CategoricalDtype(
                        categories=X_train[col].cat.categories))
                    if 'NA' not in X_test[col].cat.categories:
                        X_test[col] = X_test[col].cat.add_categories(['NA'])
                X_test[col] = X_test[col].fillna('NA')
            else:
                X_test[col] = X_test[col].astype(str).fillna('NA')

        for col in X_test.select_dtypes(include='number').columns:
            X_test[col] = X_test[col].replace([np.inf, -np.inf], np.nan)
            mean_value = num_means.get(col, 0)  # fallback if train col missing
            X_test[col] = X_test[col].fillna(mean_value)

    # LabelEncoding
    label_encoders = {}
    for col in X_train.select_dtypes(include=['object', 'category']).columns:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        label_encoders[col] = le

        if X_test is not None and col in X_test.columns:
            X_test[col] = X_test[col].astype(str)
            X_test[col] = X_test[col].map(lambda x: le.transform([x])[
                0] if x in le.classes_ else -1)

    return X_train, X_test, label_encoders


def clean_data(X):
    X = np.nan_to_num(X, nan=0.0, posinf=np.finfo(np.float64).max, neginf=np.finfo(np.float64).min)
    X = np.clip(X, -1e10, 1e10)  # 限制数值范围
    return X

from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error
import numpy as np
import gc

def evaluate_init(xtrain, ytrain, xval, yval, xtest, ytest, param_dict, 
                 task='classification', metric='auc'):

    
    clf = get_model()
    clf.fit(xtrain, ytrain)
    
    if global_config.task == 'classification':
        num_classes = len(np.unique(ytrain))
        
        if global_config.metric == 'acc':
            # 使用准确率评估
            train_pred = clf.predict(xtrain)
            val_pred = clf.predict(xval)
            test_pred = clf.predict(xtest)
            
            train_score = accuracy_score(ytrain, train_pred)
            val_score = accuracy_score(yval, val_pred)
            test_score = accuracy_score(ytest, test_pred)
            
        elif global_config.metric == 'auc':
            # 使用AUC评估
            y_train_proba = clf.predict_proba(xtrain)
            y_val_proba = clf.predict_proba(xval)
            y_test_proba = clf.predict_proba(xtest)
            
            if num_classes == 2:
                train_score = roc_auc_score(ytrain, y_train_proba[:, 1])
                val_score = roc_auc_score(yval, y_val_proba[:, 1])
                test_score = roc_auc_score(ytest, y_test_proba[:, 1])
            else:

                train_score = roc_auc_score(ytrain, y_train_proba, 
                                          multi_class='ovr', average='macro')
                # if not np.array_equal(np.unique(yval), np.unique(ytrain)):
                #     y_val_proba, used_classes = filter_and_normalize_proba(yval, ytrain, y_val_proba, clf.classes_)
                #     yval = label_binarize(yval, classes=used_classes)
                val_score = roc_auc_score(yval, y_val_proba,
                                        multi_class='ovr', average='macro')
                test_score = roc_auc_score(ytest, y_test_proba,
                                         multi_class='ovr', average='macro')
        else:
            raise ValueError(f"不支持的分类指标: {metric}")

    
    # 清理内存
    del xtrain, ytrain, xval, yval, xtest, ytest
    gc.collect()
    
    return train_score, val_score, test_score,clf

from sklearn.preprocessing import MinMaxScaler, label_binarize
def run_octree_auc_pipeline(df_train, df_test, target_column, steps):
    random.seed(0)
    np.random.seed(0)
    task_name = global_config.task_name
    task=global_config.task
    seed = global_config.data_pre["random_state"]
    model_name = global_config.model_name

    df_train, df_test, _ = preprocess_X_for_octree(df_train, df_test)

    df_train_x, df_val_x, df_train_y, df_val_y = train_test_split(df_train.drop(
        columns=[target_column]), df_train[target_column], test_size=0.3, random_state=42)

    df_test_x, df_test_y = df_test.drop(
        columns=[target_column]), df_test[target_column]

    del df_train, df_test
    gc.collect()
    cache_dir = f"baselines/{task+global_config.other_model}/{task_name}"

    cache_fir_train = f"{cache_dir}/OCTree_train_{seed}.csv"
    cache_fir_test = f"{cache_dir}/OCTree_test_{seed}.csv"
    os.makedirs(cache_dir, exist_ok=True)

    if os.path.exists(cache_fir_train) and os.path.exists(cache_fir_test):
        print("[OCTree] Loaded cached features.")
        new_features_train = pd.read_csv(cache_fir_train, )
        new_features_test = pd.read_csv(cache_fir_test)
    else:

        # 转换为NumPy数组
        xtrain = df_train_x.to_numpy()  # 训练集特征 (n_samples_train, n_features)
        ytrain = df_train_y.to_numpy()  # 训练集标签 (n_samples_train,)
        xval = df_val_x.to_numpy()  # 验证集特征 (n_samples_val, n_features)
        yval = df_val_y.to_numpy()  # 验证集标签 (n_samples_val,)
        xtest = df_test_x.to_numpy()  # 测试集特征 (n_samples_test, n_features)
        ytest = df_test_y.to_numpy()  # 测试集标签 (n_samples_test,)
        train_auc_list = []
        score_list = []
        test_auc_list = []
        r_list = []
        dt_list = []

        _, best_val, best_test,best_clf = evaluate_init(
            xtrain, ytrain, xval, yval, xtest, ytest, None)

        print("Step 0 | Val: {:.2f} | Test: {:.2f}".format(
            best_val * 100, best_test * 100))

        # Train initial predictor
        best_val = 0
        for i in range(1, 10):
            clf = XGBClassifier(
                max_depth=i,
                tree_method='hist',
                random_state=0,
                seed=0,
                eval_metric='auc'  # 可选：XGBoost 内部计算 AUC
            )
            clf = get_model()
            clf.fit(xtrain, ytrain)

            # 获取类别数量
            num_classes = len(np.unique(ytrain))

            # 获取预测概率（兼容二分类和多分类）
            ytrain_proba = clf.predict_proba(xtrain)
            yval_proba = clf.predict_proba(xval)
            ytest_proba = clf.predict_proba(xtest)
        

            # 计算AUC（自动适应二分类/多分类）
            if num_classes == 2:
                # 二分类取正类概率
                train_auc = roc_auc_score(ytrain, ytrain_proba[:, 1]) * 100
                val_auc = roc_auc_score(yval, yval_proba[:, 1]) * 100
                test_auc = roc_auc_score(ytest, ytest_proba[:, 1]) * 100
            else:
                # all_labels=np.unique(np.array(ytrain))
                # 多分类使用macro平均
                train_auc = roc_auc_score(ytrain, ytrain_proba, 
                                        multi_class='ovr', 
                                        average='macro') * 100
#                 if not np.array_equal(np.unique(yval), np.unique(ytrain)):
#                     yval_proba, used_classes = filter_and_normalize_proba(yval, ytrain, yval_proba, clf.classes_)
#                     yval = label_binarize(yval, classes=used_classes)
                    
                val_auc = roc_auc_score(yval, yval_proba,
                                      multi_class='ovr',
                                      average='macro') * 100
                test_auc = roc_auc_score(ytest, ytest_proba,
                                       multi_class='ovr',
                                       average='macro') * 100

            # 更新最佳模型
            if val_auc > best_val:
                best_train_auc, best_val, best_test = train_auc, val_auc, test_auc
                best_clf = copy.deepcopy(clf)
        importance = best_clf.feature_importances_

        sorting_imp = np.argsort(-importance)
        r0 = "x{:.0f} = [x{:.0f} * x{:.0f}]".format(
            len(xtrain[0]) + 1, sorting_imp[0] + 1, sorting_imp[1] + 1)

        def rule_template(rule):
            variables = "x1"
            for i in range(1, len(xtrain[0])):
                variables += ", x{:.0f}".format(i + 1)
            target_variable = "x{:.0f}".format(len(xtrain[0]) + 1)
            text = f'''
import numpy as np

def rule(data):
    [{variables}] = data
    {rule}
    return {target_variable}
                    '''
            return text

        rule_text = rule_template(r0)

        exec(rule_text, globals())

        new_col = [(rule(xtrain[i])) for i in range(len(xtrain))]
        new_col += [(rule(xval[i])) for i in range(len(xval))]
        new_col += [(rule(xtest[i])) for i in range(len(xtest))]
        train_auc, val_auc, test_auc = evaluate(
            new_col, xtrain, ytrain, xval, yval, xtest, ytest, None)
        best_CART = get_cart(new_col, xtrain, ytrain,
                             xval, yval, 0)  # Train CART
        dt0 = tree_to_code(best_CART, ['x{}'.format(
            i) for i in range(1, len(xtrain[0]) + 2)])  # Tree to Text
        # append
        r_list.append(r0)
        score_list.append(val_auc)
        test_auc_list.append(test_auc)
        dt_list.append(dt0)
        train_auc_list.append(train_auc)
        pattern = r"\[x{}\s*=\s*(.*?)\]".format(len(xtrain[0]) + 1)

        def check_parentheses_balanced(s):
            stack = []
            for c in s:
                if c in "([{":
                    stack.append(c)
                elif c in ")]}":
                    if not stack:
                        return False
                    if c == ")" and stack[-1] != "(":
                        return False
                    if c == "]" and stack[-1] != "[":
                        return False
                    if c == "}" and stack[-1] != "{":
                        return False
                    stack.pop()
            return len(stack) == 0

        new_features_train = pd.DataFrame()
        new_features_test = pd.DataFrame()
        # Optimize start
        for step in range(steps):
            prompt = gen_prompt(r_list, dt_list, score_list, len(xtrain[0]) + 1)

            all_status = 0
            while all_status <=5:
                answer_temp1 = use_api(prompt, None, None, 1.0)

                for num_iter in range(len(answer_temp1)):

                    if not check_parentheses_balanced(answer_temp1[num_iter]):
                        print("check_parentheses_balanced fail.")
                        continue
                        # answer_temp1[num_iter] = ss
                    match = re.search(
                        pattern, answer_temp1[num_iter], re.DOTALL)
                    if not match:
                        match = re.search(
                        pattern, "["+str(answer_temp1[num_iter])+"]", re.DOTALL)

                    if match:
                        rule_rhs = match.group(1).strip()

                        extracted_text = f"x{len(xtrain[0]) + 1} = {rule_rhs}"

                        rule_text = rule_template(extracted_text)
                        try:
                            exec(rule_text, globals())
                            # print(rule_text)

                            new_col = [(rule(xtrain[i]))
                                       for i in range(len(xtrain))]
                            new_col += [(rule(xval[i])) for i in range(len(xval))]
                            new_col += [(rule(xtest[i]))
                                        for i in range(len(xtest))]
                        except Exception:
                            print("rule error.")
                            continue

                        new_col = clean_data(new_col)

                        train_auc, val_auc, test_auc = evaluate(
                            new_col, xtrain, ytrain, xval, yval, xtest, ytest, None)
                        best_CART = get_cart(
                            new_col, xtrain, ytrain, xval, yval, seed)  # Train CART
                        dt = tree_to_code(best_CART, ['x{}'.format(
                            i) for i in range(1, len(xtrain[0]) + 2)])

                        if val_auc > np.max(np.array(score_list)):
                            new_col_train, new_col_val, new_col_test = add_column(
                                xtrain, xval, xtest, new_col)
                            new_col_name = str(len(xtrain[0]) + 1 + step)
                            new_features_train[new_col_name] = new_col_train.flatten()
                            new_features_test[new_col_name] = new_col_test.flatten()
                        elif val_auc == np.max(np.array(score_list)):
                            idxes = np.where(
                                np.array(score_list) == val_auc)[0]
                            if train_auc < np.min(np.array(train_auc_list)[idxes]):
                                new_col_train, new_col_val, new_col_test = add_column(
                                    xtrain, xval, xtest, new_col)
                                new_col_name = str(len(xtrain[0]) + 1 + step)
                                new_features_train[new_col_name] = new_col_train.flatten()
                                new_features_test[new_col_name] = new_col_test.flatten()

                        r_list.append(extracted_text)
                        score_list.append(val_auc)
                        test_auc_list.append(test_auc)
                        dt_list.append(dt)
                        train_auc_list.append(train_auc)
                        all_status = 6
                    else:
                        print(answer_temp1[num_iter])
                all_status=all_status+1
            best_val = np.max(np.array(score_list))
            best_val_idx = np.where(np.array(score_list) == best_val)[0]
            if len(best_val_idx) == 1:
                best_test = np.max(np.array(test_auc_list)[best_val_idx])
            else:
                idx = np.where(np.array(train_auc_list)[best_val_idx] == np.min(
                    np.array(train_auc_list)[best_val_idx]))[0]
                best_test = np.max(np.array(test_auc_list)[best_val_idx[idx]])

            print("Step {:.0f} | Val: {:.2f} | Test: {:.2f}".format(
                step + 1, best_val * 100, best_test * 100))
        new_features_train.to_csv(
            f"{cache_dir}/OCTree_train_{seed}.csv", index=False)
        new_features_test.to_csv(
            f"{cache_dir}/OCTree_test_{seed}.csv", index=False)
        print("[OCTree] Features generated and cached.")
        del xtrain, ytrain, xval, yval, xtest, ytest
        gc.collect()

    X_train_full = pd.concat([df_train_x.reset_index(drop=True), new_features_train], axis=1)
    X_test_full = pd.concat([df_test_x.reset_index(drop=True), new_features_test], axis=1)
    
    X_train_full,X_test_full,_=preprocess_X(X_train_full,X_test_full)
    train_score,test_score=evaluate_model(X_train_full,df_train_y,X_test_full,df_test_y)
    del df_test_x, df_test_y, df_train_x, df_train_y, df_val_x, df_val_y, X_train_full, X_test_full, new_features_train, new_features_test
    gc.collect()
    
    return train_score,test_score
