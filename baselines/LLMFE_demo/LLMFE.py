import os
import json
import numpy as np
import pandas as pd
from sklearn import preprocessing
from .llmfe import config
from .llmfe import sampler
from .llmfe import evaluator
from .llmfe import pipeline
from .utils import is_categorical


def extract_description_field(description: dict) -> dict:
    """
    从嵌套字典中提取每个字段的 description 字段。

    参数：
        description (dict): 原始的描述信息，包含嵌套结构

    返回：
        meta_data (dict): 扁平化后的变量名 -> 描述 映射
    """
    meta_data = {
        key: value["description"]
        for key, value in description.items()
        if isinstance(value, dict) and "description" in value
    }
    return meta_data


import re


def run_llmfe_feature_engineering(
        df_train: pd.DataFrame,
        # df_test: pd.DataFrame,
        target_column: str,
        task_name: str,
        description: json,
        task_description: str,
        task:str,
        seed: int,
        file_model_name: str,
        use_api: bool = True,
        api_model: str = "deepseek-chat",
        other_model:str="",
        api_key:str="",
        max_sample_num: int = 20
):
    """
    Run LLM-FE Feature Engineering pipeline with custom data input.

    Args:
        task_name: task_name
        df_train: Training DataFrame
        target_column: The name of the target column
        use_api: Whether to use API or local LLM
        api_model: API model name
        max_sample_num: Number of LLM samples
    """
    current_file_path = os.path.dirname(os.path.abspath(__file__))
    spec_path = f"{current_file_path}/specs/specification_{task}.txt"
    # Load prompt specification
    with open(spec_path, encoding="utf-8") as f:
        specification = f.read()
    specification = re.sub(r'(<task>\n)', r'<task>\n' + task_description, specification)


    # Prepare label and features
    label_encoder = preprocessing.LabelEncoder()
    is_regression=False
    
    if task == "regression":
        is_regression = True
    # print(df_train.columns)

    X_train = df_train.drop(columns=[target_column])
    # X_test = df_test.drop(columns=[target_column])
    y_train = df_train[target_column].values
    # y_test = df_test[target_column].values
    label_list = np.unique(y_train).tolist()
    # Handle categorical encoding
    for col in X_train.columns:
        if X_train[col].dtype == 'string' or X_train[col].dtype == object:
            le = preprocessing.LabelEncoder()
            full_col = pd.concat([X_train[col]]).astype(str)
            le.fit(full_col)
            X_train[col] = le.transform(X_train[col].astype(str))
            # X_test[col] = le.transform(X_test[col].astype(str))

    X_train = X_train.fillna(0)
    # X_test = X_test.fillna(0)

    if not is_regression:
        y_train = label_encoder.fit_transform(y_train)
        # y_test = label_encoder.transform(y_test)

    meta_data = extract_description_field(description)
    is_cat = [is_categorical(X_train.iloc[:, i]) for i in range(X_train.shape[1])]

    # Prepare data dict for pipeline
    data_dict = {
        'inputs': X_train,
        'outputs': y_train,
        'is_cat': is_cat,
        'is_regression': is_regression
    }
    dataset = {'data': data_dict}

    # Configure LLM-FE pipeline
    class_config = config.ClassConfig(llm_class=sampler.LocalLLM, sandbox_class=evaluator.LocalSandbox)
    cfg = config.Config(use_api=use_api, api_model=api_model,api_key=api_key)

    pipeline.main(
        specification=specification,
        inputs=dataset,
        config=cfg,
        meta_data=meta_data,
        max_sample_nums=max_sample_num,
        class_config=class_config,
        log_dir=f"{current_file_path}/logs",
        save_final_code=True,
        final_code_path=f"baselines/{task+other_model}/{task_name}/LLMFE_final_modify_features_{seed}.py"
    )

    # print(f"Feature generation completed. Final function saved to {log_path}/final_modify_features.py")
