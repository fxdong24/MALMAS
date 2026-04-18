"""
Perform Feature Engineering
"""
# Imports
import os
import json
from argparse import ArgumentParser
import numpy as np
import pandas as pd
from utils import is_categorical
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold, KFold

# Arguments
parser = ArgumentParser()
parser.add_argument('--port', type=int, default=None)
parser.add_argument('--use_api', type=bool, default=False)
parser.add_argument('--api_model', type=str, default="deepseek-chat")
parser.add_argument('--spec_path', type=str)
parser.add_argument('--log_path', type=str, default="./logs/oscillator1")
parser.add_argument('--problem_name', type=str, default="oscillator1")
parser.add_argument('--run_id', type=int, default=1)
args = parser.parse_args()


if __name__ == '__main__':
    # Define the maximum number of iterations
    global_max_sample_num = 20
    splits = 5
    seed = 42
    # Load prompt specification
    with open(
        os.path.join(args.spec_path),
        encoding="utf-8",
    ) as f:
        specification = f.read()

    problem_name = args.problem_name
    label_encoder = preprocessing.LabelEncoder()
    is_regression = False
    if problem_name in ['forest-fires', 'housing', 'insurance', 'bike', 'wine', 'crab']:
        is_regression = True

    # Load data observations
    file_name = f"./data/{problem_name}.csv"
    df = pd.read_csv(file_name)
    
    target_attr = df.columns[-1]
    is_cat = [is_categorical(df.iloc[:, i]) for i in range(df.shape[1])][:-1]
    attribute_names = df.columns[:-1].tolist()

    X = df.convert_dtypes()
    y = df[target_attr].to_numpy()
    label_list = np.unique(y).tolist()

    X = X.drop(target_attr, axis=1)

    for col in X.columns:
        if X[col].dtype == 'string':
            X[col] = label_encoder.fit_transform(X[col])


    # Handle missing values
    X = X.fillna(0)
    if is_regression == False:
        y = label_encoder.fit_transform(y)
    else:
        y = y
 
    # Load metadata
    meta_data_name = f"./data/{problem_name}-metadata.json"
    meta_data={}
    try:
        with open(meta_data_name, "r") as f:
            filed_meta_data = json.load(f)
    except:
        filed_meta_data = {}
    meta_data = dict(meta_data, **filed_meta_data)
    
    skf = KFold(n_splits=splits, shuffle=True, random_state=42) if is_regression else StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)

    i = 0
    for train_idx, test_idx in skf.split(X, y):
        # Load config and parameters
        from llmfe import config
        from llmfe import sampler
        from llmfe import evaluator
        from llmfe import pipeline

        class_config = config.ClassConfig(llm_class=sampler.LocalLLM, sandbox_class=evaluator.LocalSandbox)
        config = config.Config(use_api = args.use_api,
                            api_model = args.api_model,)
        X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
        y_train_fold, y_test_fold = y[train_idx], y[test_idx]
        i +=1

        data_dict = {'inputs': X_train_fold, 'outputs': y_train_fold, 'is_cat': is_cat, 'is_regression': is_regression}
        dataset = {'data': data_dict}
        log_path = args.log_path + f"_split_{i}"

        pipeline.main(
            specification=specification,
            inputs=dataset,
            config=config,
            meta_data=meta_data,
            max_sample_nums=global_max_sample_num*splits,
            class_config=class_config,
            log_dir = f'logs/llama3/{log_path}',
        )