from .path_helper import add_base_to_sys_path
add_base_to_sys_path(2)
import openai
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import datetime
from xgboost import XGBRegressor,XGBClassifier
from .model_factory import get_model
from sklearn.metrics import mean_squared_error,roc_auc_score,accuracy_score
import gc
from datetime import datetime as dt_class
import math
import time
import global_config
import sys
import re
import json
from typing import List, Dict,Optional
import importlib
from tqdm import tqdm
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.model_selection import cross_val_predict
from scipy.stats import ks_2samp, chi2_contingency,gaussian_kde

def process_new_features_list(new_features_list):

    concatenated_features = pd.concat([i for i in new_features_list], axis=1)
    

    duplicated_columns = concatenated_features.columns[concatenated_features.columns.duplicated(keep=False)]

    unique_features = concatenated_features.loc[:, ~concatenated_features.columns.duplicated(keep='first')]
    
    return unique_features

def get_xgboost_feature_importance(df, target_column, top_rate=1, importance_type='gain'):
    if len(df.columns)>20:
        top_rate=0.4
    elif len(df.columns)>10:
        top_rate=0.6
    else:
        top_rate=1.0
        
    df, _, _ = preprocess_X(df)
    X = df.drop(columns=[target_column])
    y = df[target_column]
    model = get_model()
    model.fit(X, y)

    booster = model.get_booster()
    score = booster.get_score(importance_type=importance_type)
    importance_df = pd.DataFrame([
        {'feature': k, 'importance': v} for k, v in score.items()
    ])
    total = importance_df['importance'].sum()
    importance_df['importance_percent'] = 100 * \
        importance_df['importance'] / total

    importance_df = importance_df.sort_values(
        by='importance_percent', ascending=False)

    importance_df = importance_df.drop(
        columns="importance").reset_index(drop=True)

    top_k = int(len(df.columns)*top_rate+0.5)
    lines = [
        f"{i+1}. {row['feature']}: {row['importance_percent']:.2f}%"
        for i, row in importance_df.head(top_k).iterrows()
    ]

    importance_str = "\n".join(lines)
    return importance_str

import copy

def persist_top_features_and_update_description(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    agent_gains: dict,  
    agent_features: dict,  
    description: dict,  
    enrich_description: dict,  
    target_column: str = None,  
    top_k: int = 5,  
):


    new_columns_this_round = []

    for agent_name, gain_df in agent_gains.items():
        if gain_df.empty:
            continue
        
        top_features = gain_df.sort_values(by="gain", ascending=False).head(top_k)

        for _, row in top_features.iterrows():
            fname = row["feature"]
            if fname not in agent_features[agent_name]["train_positive"].columns:
                continue
            if fname in df_train.columns:
                continue
            df_train = pd.concat([df_train, agent_features[agent_name]["train_positive"][[fname]]], axis=1)
            df_test = pd.concat([df_test, agent_features[agent_name]["test_positive"][[fname]]], axis=1)

            description[fname] = {
                "description": row.get("logic", "No description available"),
                "type": row.get("type", "unknown")
            }

            new_columns_this_round.append(fname)

    # enrich 只针对新增列进行
    if new_columns_this_round:
        enriched_new = enrich_field_info_for_local_pattern(
            df_train[new_columns_this_round],
            {k: description[k] for k in new_columns_this_round},
            target_col=target_column
        )
        enrich_description.update(enriched_new)

    return df_train, df_test, description, enrich_description


import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
from sklearn.preprocessing import KBinsDiscretizer

def enrich_field_info_for_local_pattern(df: pd.DataFrame, field_info: dict, target_col: str = None):

    enriched_info = {}

    for col, info in field_info.items():
        if col not in df.columns:
            continue
        if col == "Id":
            enriched_info[col] = info 
            continue

        col_data = df[col]

        if isinstance(col_data, pd.DataFrame):
            col_data = col_data.squeeze(axis=1)

        if not isinstance(col_data, pd.Series):
            raise ValueError(f"Expected Series for column '{col}', but got {type(col_data)}")

        info = info.copy()  

        info["missing_ratio"] = float(col_data.isna().mean())
        if info.get("type") == "numerical":
            clean_data = col_data.dropna()
            info["min"] = float(clean_data.min())
            info["max"] = float(clean_data.max())
            info["mean"] = float(clean_data.mean())
            info["std"] = float(clean_data.std())
            info["skewness"] = float(clean_data.skew())
            info["kurtosis"] = float(clean_data.kurt())
            info["outlier_ratio_3sigma"] = float(
                ((clean_data - clean_data.mean()).abs() > 3 * clean_data.std()).mean())

            try:
                kde = gaussian_kde(clean_data)
                grid = np.linspace(clean_data.min(), clean_data.max(), 1000)
                density = kde(grid)
                peaks = np.diff(np.sign(np.diff(density))) < 0
                num_modes = int(peaks.sum())
                info["num_modes_estimated"] = num_modes
            except Exception:
                info["num_modes_estimated"] = None

        elif info.get("type") == "categorical":
            value_counts = col_data.value_counts(dropna=False)
            info["n_unique"] = int(col_data.nunique(dropna=True))
            info["values"] = value_counts.index.astype(str).tolist()
            info["value_frequencies"] = value_counts.values.tolist()

        if target_col and target_col in df.columns:
            try:
                target_data = df[[col, target_col]].dropna()
                if target_data.shape[0] >= 10:
                    est = KBinsDiscretizer(
                        n_bins=10, encode='ordinal', strategy='quantile')
                    bins = est.fit_transform(target_data[[col]])
                    target_data = target_data.copy()  
                    target_data['bin'] = bins.astype(int)
                    bin_means = target_data.groupby(
                        'bin')[target_col].mean().tolist()
                    info["target_bin_avg"] = bin_means
            except Exception:
                info["target_bin_avg"] = None

        enriched_info[col] = info  # 加入增强结果

    return enriched_info


def remove_duplicate_columns(df):
    # 找出重复的列
    duplicate_cols = df.columns[df.columns.duplicated()]
    
    if len(duplicate_cols) > 0:
        df = df.loc[:, ~df.columns.duplicated(keep='first')]
    return df

def preprocess_X(X_train, X_test=None, drop_cols=None):

    X_train = X_train.copy()
    X_train=remove_duplicate_columns(X_train)

    drop_cols = drop_cols or []

    # 删除指定列
    X_train = X_train.drop(columns=drop_cols, errors='ignore')
    if X_test is not None:
        X_test = X_test.copy()
        X_test = X_test.drop(columns=drop_cols, errors='ignore')
        X_test=remove_duplicate_columns(X_test)

    # 填充训练集
    for col in X_train.select_dtypes(include=['object', 'category']).columns:
        if pd.api.types.is_categorical_dtype(X_train[col]):
            if 'NA' not in X_train[col].cat.categories:
                X_train[col] = X_train[col].cat.add_categories(['NA'])
        X_train[col] = X_train[col].fillna('NA')

    for col in X_train.select_dtypes(include='number').columns:
        X_train[col] = X_train[col].replace([np.inf, -np.inf], np.nan).fillna(0)

    # 填充测试集
    if X_test is not None:
        for col in X_test.select_dtypes(include=['object', 'category']).columns:
            if col in X_train.columns:
                # 如果训练集有这个列，保持测试集列的类别和训练集一致
                if pd.api.types.is_categorical_dtype(X_train[col]):
                    X_test[col] = X_test[col].astype(pd.CategoricalDtype(
                        categories=X_train[col].cat.categories
                    ))
                    # 如果'NA'不在类别中，再加进去
                    if 'NA' not in X_test[col].cat.categories:
                        X_test[col] = X_test[col].cat.add_categories(['NA'])
                X_test[col] = X_test[col].fillna('NA')
            else:
                # 如果训练集中没有该列，直接转成object后填充
                X_test[col] = X_test[col].astype(str).fillna('NA')

        for col in X_test.select_dtypes(include='number').columns:
            X_test[col] = X_test[col].replace([np.inf, -np.inf], np.nan).fillna(0)

            

    # LabelEncoding
    label_encoders = {}
    for col in X_train.select_dtypes(include=['object', 'category']).columns:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        label_encoders[col] = le

        if X_test is not None and col in X_test.columns:
            X_test[col] = X_test[col].astype(str)

            # 对未见类别用 -1 编码
            X_test[col] = X_test[col].map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

    return X_train, X_test, label_encoders



import time

def generate_response(model_name, key, url, systemprompt, prompt, temperature, 
                     assistant_content=None, userprompt2=None):

    client = openai.OpenAI(
        api_key=key,
        base_url=url
    )

    # 每次把完整历史拼成 messages
    messages = [{"role": "system", "content": systemprompt}]

    # 任务描述
    messages.append({"role": "user", "content": prompt})

    # 如果是纠错流程，明确表达「上次代码错误」：
    if assistant_content and userprompt2:
        messages.append({"role": "assistant", "content": assistant_content})
        messages.append({"role": "user", "content": f"""
    Your last code has an error:

    {userprompt2}

    Please revise your previous code and output a corrected version. Only output code, no explanation.
    """} )

    try:

        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            stream=False,
            temperature=temperature
        )

    except Exception as e:
        print("出错了:\n"+str(e))
        return None
    if global_config.compute_tokens:
        global_config.total_tokens = global_config.total_tokens + response.usage.total_tokens

    return response.choices[0].message.content


def extract_and_execute_function(code_output: str):


    code_blocks = re.findall(r"```(?:python)?(.*?)```", code_output, re.DOTALL)
    if not code_blocks:
        code_blocks = re.findall(r"(def\s+\w+\s*\(.*?\)\s*:[\s\S]+?)(?=\n\S|$)", code_output)
    if not code_blocks:
        raise ValueError("❌ ")

    code = code_blocks[0].strip()
    match = re.search(r"def\s+(\w+)\s*\(", code)
    if not match:
        raise ValueError("❌ ")
    func_name = match.group(1)

    injected_globals = {
        'pd': pd,
        'np': np,
        'datetime': dt_class,
        'datetime_module': datetime,
        'math': math,
        'time': time,
        're': re,
    }
    import_lines = re.findall(r'^\s*(import\s+[^\n]+|from\s+[^\n]+import\s+[^\n]+)', code, re.MULTILINE)

    for line in import_lines:
        try:
            if line.startswith("import "):
                modules = line.replace("import", "").strip().split(",")
                for mod in modules:
                    mod = mod.strip()
                    if " as " in mod:
                        module_name, alias = map(str.strip, mod.split(" as "))
                        injected_globals[alias] = importlib.import_module(module_name)
                    else:
                        injected_globals[mod] = importlib.import_module(mod)
            elif line.startswith("from "):
                match = re.match(r'from\s+([\w\.]+)\s+import\s+(.+)', line)
                if match:
                    mod_name, imports = match.groups()
                    module = importlib.import_module(mod_name)
                    for item in imports.split(","):
                        item = item.strip()
                        if " as " in item:
                            obj_name, alias = map(str.strip, item.split(" as "))
                            injected_globals[alias] = getattr(module, obj_name)
                        else:
                            injected_globals[item] = getattr(module, item)
        except Exception as e:
            print(f"⚠️ 依赖注入失败: `{line}` -> {e}")

    # === Step 5: 执行代码 ===
    local_vars = {}
    try:
        exec(code, injected_globals, local_vars)
    except Exception as e:
        raise RuntimeError(f"❌ llm-codeSyntaxError: invalid syntax。")

    # === Step 6: 返回函数 ===
    if func_name not in local_vars:
        raise ValueError(f"❌ `{func_name}` ")
    
    return local_vars[func_name], code

def evaluate_new_feature_gain_cv(
    base_features_df: pd.DataFrame,
    new_features_df: pd.DataFrame,
    target_column: str,
    n_splits: int = 5,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Evaluates each new feature's impact on model RMSE using K-fold CV.
    
    Parameters:
        base_features_df : pd.DataFrame
            Original features including target column
        new_features_df : pd.DataFrame
            Candidate new features (one per column)
        target_column : str
            Name of target variable column
        n_splits : int
            Number of cross-validation folds
        verbose : bool
            Whether to print progress information
            
    Returns:
        pd.DataFrame: RMSE gains for each new feature, sorted by gain
    """
    # Get baseline performance
    base_rmse_train, base_rmse_val = train_cv_Regressor(
        df=base_features_df,
        target_column=target_column,
        n_splits=n_splits
    )

    results = []
    
    # Evaluate each new feature
    for col in new_features_df.columns:
        combined_df = pd.concat([base_features_df, new_features_df[[col]]], axis=1)
        
        new_rmse_train, new_rmse_val = train_cv_Regressor(
            df=combined_df,
            target_column=target_column,
            n_splits=n_splits
        )
        
        gain = base_rmse_val - new_rmse_val
        
        if verbose:
            print(f"[{col}] base RMSE: {base_rmse_val:.5f} → new RMSE: {new_rmse_val:.5f} | Gain: {gain:.5f}")
        
        results.append({
            "feature": col,
            "base_rmse_val": base_rmse_val,
            "new_rmse_val": new_rmse_val,
            "gain": gain
        })

    # Return sorted results
    return pd.DataFrame(results).sort_values("gain", ascending=False).reset_index(drop=True)




def evaluate_new_feature_gain_cv_cls(
    base_features_df: pd.DataFrame,
    new_features_df: pd.DataFrame,
    target_column: str,
    n_splits: int = 5,
    verbose: bool = True
) -> pd.DataFrame:

    # 获取基础模型的AUC和准确率
    base_auc_train, base_auc_val, base_acc_train, base_acc_val = train_cv_Classifier(
        df=base_features_df,
        target_column=target_column,
        n_splits=n_splits
    )

    results = []

    for col in new_features_df.columns:
        # 合并基础特征和新特征
        combined_df = pd.concat([base_features_df, new_features_df[[col]]], axis=1)

        # 获取加入新特征后的模型性能
        new_auc_train, new_auc_val, new_acc_train, new_acc_val = train_cv_Classifier(
            df=combined_df,
            target_column=target_column,
            n_splits=n_splits
        )

        # 计算综合增益（AUC增益 + 准确率增益）
        auc_gain = new_auc_val - base_auc_val
        acc_gain = new_acc_val - base_acc_val
        total_gain = auc_gain + acc_gain

        if verbose:
            print(f"[{col}] "
                  f"base AUC: {base_auc_val:.5f} → new AUC: {new_auc_val:.5f} | AUC gain: {auc_gain:.5f} | "
                  f"base ACC: {base_acc_val:.5f} → new ACC: {new_acc_val:.5f} | ACC gain: {acc_gain:.5f} | "
                  f"gain: {total_gain:.5f}")

        results.append({
            "feature": col,
            "base_auc_val": base_auc_val,
            "new_auc_val": new_auc_val,
            "auc_gain": auc_gain,
            "base_acc_val": base_acc_val,
            "new_acc_val": new_acc_val,
            "acc_gain": acc_gain,
            "gain": total_gain  
        })

    return pd.DataFrame(results).sort_values("gain", ascending=False).reset_index(drop=True)

from sklearn.metrics import accuracy_score 

def train_cv_Classifier(
    df,
    target_column,
    n_splits=5,
):

    df = df.copy()
    df, _, _ = preprocess_X(df)
    X = df.drop(columns=[target_column])
    y = df[target_column]

    

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    auc_trains = []
    auc_vals = []
    acc_trains = []  
    acc_vals = []    

    for fold, (cv_train_idx, cv_val_idx) in enumerate(kf.split(X)):
        X_cv_train, X_cv_val = X.iloc[cv_train_idx], X.iloc[cv_val_idx]
        y_cv_train, y_cv_val = y.iloc[cv_train_idx], y.iloc[cv_val_idx]

        model = get_model()

        if not hasattr(model, 'fit') or not (hasattr(model, 'predict_proba') or hasattr(model, 'decision_function')):
            raise TypeError("get_model() 返回的模型不具备 fit 和 predict_proba/decision_function 方法")
        if not df.columns.is_unique:
            print(f"\n\ndf.columns.is_unique:{X_cv_train.columns.is_unique}\n\n")
        model.fit(X_cv_train, y_cv_train)

        y_train_pred = model.predict(X_cv_train)
        y_val_pred = model.predict(X_cv_val)

        acc_train = accuracy_score(y_cv_train, y_train_pred)
        acc_val = accuracy_score(y_cv_val, y_val_pred)
        acc_trains.append(acc_train)
        acc_vals.append(acc_val)

        if hasattr(model, "predict_proba"):
            y_train_score = model.predict_proba(X_cv_train)
            y_val_score = model.predict_proba(X_cv_val)
        else:
            y_train_score = model.decision_function(X_cv_train)
            y_val_score = model.decision_function(X_cv_val)

        if len(np.unique(y_cv_train)) == 2: 
            auc_train = roc_auc_score(y_cv_train, y_train_score[:, 1])
            auc_val = roc_auc_score(y_cv_val, y_val_score[:, 1])
        else: 
            auc_train = roc_auc_score(y_cv_train, y_train_score, 
                                    multi_class='ovr', 
                                    average='macro')
            auc_val = roc_auc_score(y_cv_val, y_val_score,
                                  multi_class='ovr',
                                  average='macro')

        auc_trains.append(auc_train)
        auc_vals.append(auc_val)

    return (
        np.mean(auc_trains), 
        np.mean(auc_vals),
        np.mean(acc_trains),
        np.mean(acc_vals)
    )


def train_cv_Regressor(
    df,
    target_column,
    n_splits=5,
):

    df = df.copy()
    df, _, _ = preprocess_X(df)
    X = df.drop(columns=[target_column])
    y = df[target_column]

    
    kf = KFold(n_splits=n_splits, shuffle=False,
               random_state=None)
    rmse_trains = []
    rmse_vals = []

    for fold, (cv_train_idx, cv_val_idx) in enumerate(kf.split(X)):
        X_cv_train, X_cv_val = X.iloc[cv_train_idx], X.iloc[cv_val_idx]
        y_cv_train, y_cv_val = y.iloc[cv_train_idx], y.iloc[cv_val_idx]

        model = get_model()

        # 检查是否是回归模型
        if not hasattr(model, 'predict') or not hasattr(model, 'fit'):
            raise TypeError("get_model() 返回的模型不具备 fit/predict 方法，可能不是回归模型。")

        model.fit(X_cv_train, y_cv_train)

        y_pred_train = model.predict(X_cv_train)
        y_pred_val = model.predict(X_cv_val)
        rmse_train_val = nrmse(y_cv_train, y_pred_train)
        rmse_val = nrmse(y_cv_val, y_pred_val)

        rmse_trains.append(rmse_train_val)
        rmse_vals.append(rmse_val)

    avg_rmse_train = np.mean(rmse_trains)
    avg_rmse_val = np.mean(rmse_vals)

    return avg_rmse_train, avg_rmse_val

def nrmse(y_true,y_pred):
    rmse=np.sqrt(mean_squared_error(y_true, y_pred))
    norm=np.mean(y_true)
    return rmse/norm

def test_Regressor(
    df_train,
    df_test,
    target_column,
):


    df_train = df_train.copy()
    df_test = df_test.copy()
    df_train, df_test, _ = preprocess_X(df_train, df_test)
    # 分离特征与目标
    X_train = df_train.drop(columns=[target_column])
    y_train = df_train[target_column]
    X_test = df_test.drop(columns=[target_column])
    y_test = df_test[target_column]


    # 训练模型
    model = get_model()
    model.fit(X_train, y_train)

    # 预测并计算 RMSE
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    rmse_train = nrmse(y_train, y_pred_train)
    rmse_test = nrmse(y_test, y_pred_test)

    return model, rmse_train, rmse_test
def main_func_evaluate_model(train_X, train_y, test_X, test_y):

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
    return model,train_score, test_score


def test_Classifier(
    df_train,
    df_test,
    target_column,
):

    if global_config.model_name=="tpfn" and df_train.shape[0]>1000:
        df_train=df_train.sample(n=1000, random_state=42)
        df_test=df_test.sample(n=1000, random_state=42)
    df_train = df_train.copy()
    df_test = df_test.copy()
    df_train, df_test, _ = preprocess_X(df_train, df_test)
 
    X_train = df_train.drop(columns=[target_column])
    y_train = df_train[target_column]
    X_test = df_test.drop(columns=[target_column])
    y_test = df_test[target_column]
    
        

    return main_func_evaluate_model(X_train,y_train,X_test,y_test)



def classification_adv_validation_df(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,

    new_feature_train: pd.DataFrame,
    new_feature_test: pd.DataFrame,
    target_col: str,
    p_thresh: float = 0.01,
    auc_thresh: float = 0.65,
    n_folds: int = 5
) -> pd.DataFrame:

    train_df, test_df, _ = preprocess_X(train_df, test_df)
    
    X_train=train_df.drop(columns=[target_column])
    X_test=test_df.drop(columns=[target_column])
    y_train = train_df[target_col]
    y_test = test_df[target_col]

    if new_feature_train is not None:
        new_feature_train, new_feature_test, _ = preprocess_X(
            new_feature_train, new_feature_test)

    # 合并新特征
    if new_feature_train is not None:
        X_train = pd.concat([X_train, new_feature_train], axis=1)
        X_test = pd.concat([X_test, new_feature_test], axis=1)
        feature_sources = {**{col: 'original' for col in X_train.columns[:len(train_df.columns)-1]},
                           **{col: 'new' for col in new_feature_train.columns}}
    else:
        feature_sources = {col: 'original' for col in X_train.columns}

    # 2. 对抗验证
    X_combined = pd.concat([X_train, X_test])
    y_adv = np.array([0]*len(X_train) + [1]*len(X_test))

    model = XGBClassifier(n_estimators=100, random_state=42, eval_metric='auc')
    preds = cross_val_predict(model, X_combined, y_adv,
                              cv=n_folds, method='predict_proba')[:, 1]
    global_auc = roc_auc_score(y_adv, preds)

    # 3. 特征级别分析
    results = []
    for col in tqdm(X_combined.columns, desc="Analyzing Features"):
        # 数值型特征检验
        ks_stat, ks_p = (np.nan, np.nan)
        if pd.api.types.is_numeric_dtype(X_combined[col]):
            ks_stat, ks_p = ks_2samp(
                X_train[col].dropna(), X_test[col].dropna())

        # 类别型特征检验
        chi2_p = np.nan
        if not pd.api.types.is_numeric_dtype(X_combined[col]):
            contingency = pd.crosstab(
                pd.concat([X_train[col], X_test[col]]),
                y_adv
            )
            _, chi2_p, _, _ = chi2_contingency(contingency)

        # 单特征AUC
        single_model = XGBClassifier(n_estimators=50, random_state=42)
        single_preds = cross_val_predict(single_model,
                                         X_combined[[col]],
                                         y_adv,
                                         cv=3,
                                         method='predict_proba')[:, 1]
        feat_auc = roc_auc_score(y_adv, single_preds)

        # 判断逻辑
        keep = True
        reason = None
        if not pd.isna(ks_p) and (ks_p < p_thresh and ks_stat > 0.25):
            keep = False
            reason = f"KS(p={ks_p:.1e},D={ks_stat:.2f})"
        elif not pd.isna(chi2_p) and chi2_p < p_thresh:
            keep = False
            reason = f"Chi2(p={chi2_p:.1e})"
        elif feat_auc > auc_thresh:
            keep = False
            reason = f"AUC={feat_auc:.2f}"

        results.append({
            'feature': col,
            'source': feature_sources[col],
            'ks_stat': ks_stat,
            'ks_p': ks_p,
            'chi2_p': chi2_p,
            'auc_score': feat_auc,
            'should_keep': keep,
            'reason': reason
        })

    # 4. 整理结果
    result_df = pd.DataFrame(results)
    result_df['global_auc'] = global_auc

    adv_result_df = result_df.sort_values('auc_score', ascending=False)
    keep = adv_result_df[adv_result_df['should_keep']]
    drop = adv_result_df[~adv_result_df['should_keep']]
    keep_or_drop = {
        'keep_original': keep[keep['source'] == 'original']['feature'].tolist(),
        'keep_new': keep[keep['source'] == 'new']['feature'].tolist(),
        'drop_original': dict(zip(
            drop[drop['source'] == 'original']['feature'],
            drop[drop['source'] == 'original']['reason']
        )),
        'drop_new': dict(zip(
            drop[drop['source'] == 'new']['feature'],
            drop[drop['source'] == 'new']['reason']
        )),
        'global_auc': adv_result_df['global_auc'].iloc[0]
    }
    drop = list(keep_or_drop["drop_new"].keys())
    model, train_auc, test_auc = test_Classifier(process_new_features_list([train_df,new_feature_train]).drop(columns=drop),
                                                 process_new_features_list([test_df, new_feature_test]).drop(columns=drop),
                                                 target_col,)
    return adv_result_df, keep_or_drop, train_auc, test_auc

def extract_feature_json(text: str) -> List[Dict]:

    start = text.find('[')
    end = text.rfind(']')

    if start == -1 or end == -1 or start >= end:
        raise ValueError("❌ 无法找到有效的 JSON 数组边界")

    json_str = text[start:end+1]

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        print("⚠️ JSON 字符串预览（前 300 字）：", json_str[:300])
        raise ValueError(f"JSON 解析失败: {e}")

    return data


from typing import Union

def find_feature_metadata(data: List[Dict], target_name: str) -> Optional[Dict]:

    for item in data:
        base = item.get("base_columns")

        # 统一转换成列表
        if isinstance(base, str):
            base_columns = [base]
        elif isinstance(base, list):
            base_columns = base
        else:
            continue  # 跳过无法识别的格式

        for feat in item.get("derived_features", []):
            if feat.get("name") == target_name:
                return {
                    "base_columns": base_columns,
                    "type":feat.get("type"),
                    "transform":feat.get("transform"),
                    "logic": feat.get("logic")
                }

    return None
