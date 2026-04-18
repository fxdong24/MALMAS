import os
import sys
from sklearn.preprocessing import KBinsDiscretizer
from scipy.stats import gaussian_kde
import pandas as pd
import numpy as np
import json
from datetime import datetime
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
def get_base_dir(levels_up=1):

    try:
        # 正常脚本中使用 __file__
        current_path = os.path.abspath(__file__)
    except NameError:
        # Jupyter 中使用当前工作路径
        current_path = os.path.abspath(os.getcwd())
        
    for _ in range(levels_up):
        current_path = os.path.dirname(current_path)
        
    return current_path


def add_base_to_sys_path(levels_up=1):

    base_path = get_base_dir(levels_up)
    if base_path not in sys.path:
        sys.path.append(base_path)
add_base_to_sys_path(4)

import global_config
from main_demo.model_factory import *
import time
import random
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.tree import _tree
from sklearn import tree
#from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

from sklearn.metrics import roc_auc_score




def add_column(xtrain, xval, xtest, gen_c):
    gen_c = np.array(gen_c).reshape(-1,1)
    len_train, len_val = len(xtrain), len(xval)
    gen_c_train, gen_c_val, gen_c_test = gen_c[:len_train], gen_c[len_train:len_train + len_val], gen_c[len_train + len_val:]
    enc = MinMaxScaler()
    enc = enc.fit(gen_c_train)
    gen_c_train = enc.transform(gen_c_train)
    gen_c_val = enc.transform(gen_c_val)
    gen_c_test = enc.transform(gen_c_test)
    return gen_c_train, gen_c_val, gen_c_test
def clean_data(X):
    X = np.nan_to_num(X, nan=0.0, posinf=np.finfo(np.float64).max, neginf=np.finfo(np.float64).min)
    X = np.clip(X, -1e10, 1e10)  # 限制数值范围
    return X


import numpy as np



def evaluate(gen_c, xtrain, ytrain, xval, yval, xtest, ytest, param_dict):
    gen_c = np.array(gen_c).reshape(-1,1)
    gen_c=clean_data(gen_c)
    len_train, len_val = len(xtrain), len(xval)
    gen_c_train, gen_c_val, gen_c_test = gen_c[:len_train], gen_c[len_train:len_train + len_val], gen_c[len_train + len_val:]
    enc = MinMaxScaler()
    enc = enc.fit(gen_c_train)
    gen_c_train = enc.transform(gen_c_train)
    gen_c_val = enc.transform(gen_c_val)
    gen_c_test = enc.transform(gen_c_test)
    new_train = np.concatenate([xtrain, gen_c_train], axis = -1)
    new_val = np.concatenate([xval, gen_c_val], axis = -1)
    new_test = np.concatenate([xtest, gen_c_test], axis = -1)
    clf = get_model()
    clf.fit(new_train, ytrain)
    num_classes = len(np.unique(ytrain))
    
    # 获取预测概率
    ytrain_proba = clf.predict_proba(new_train)
    yval_proba = clf.predict_proba(new_val)
    ytest_proba = clf.predict_proba(new_test)
    
    if num_classes == 2:
        train_auc = roc_auc_score(ytrain, ytrain_proba[:, 1])
        val_auc = roc_auc_score(yval, yval_proba[:, 1])
        test_auc = roc_auc_score(ytest, ytest_proba[:, 1])
    else:
        all_labels=np.unique(np.array(ytrain))
        # 多分类处理
        train_auc = roc_auc_score(ytrain, ytrain_proba, 
                                multi_class='ovr', 
                                average='macro')
        val_auc = roc_auc_score(yval, yval_proba,labels=all_labels,
                              multi_class='ovr',
                              average='macro')
        test_auc = roc_auc_score(ytest, ytest_proba,
                               multi_class='ovr',
                               average='macro')
    
    return train_auc, val_auc, test_auc
from sklearn.preprocessing import MinMaxScaler, label_binarize

import numpy as np

def filter_and_normalize_proba(yval, ytrain, yval_proba, classes_in_model):
    """
    根据 yval 的标签，过滤 yval_proba 只保留 yval 里出现的类别，并归一化概率。
    
    参数:
    - yval: 验证集标签
    - ytrain: 训练集标签（用于判断是否全量）
    - yval_proba: 验证集的预测概率，shape (n_samples, n_classes)
    - classes_in_model: clf.classes_，表示 yval_proba 的列对应的类别
    
    返回:
    - filtered_proba: 过滤后并归一化的概率
    - used_classes: 保留的类别
    """
    yval_labels = np.unique(yval)
    ytrain_labels = np.unique(ytrain)

    if np.array_equal(yval_labels, ytrain_labels):
        # 没有问题，直接返回原始概率
        return yval_proba, classes_in_model
    else:
        # 过滤出 yval 中存在的列
        mask = np.isin(classes_in_model, yval_labels)
        filtered_proba = yval_proba[:, mask]
        
        row_sums = np.sum(filtered_proba, axis=1, keepdims=True)
        
        row_sums[row_sums == 0] = 1.0
        normalized_proba = filtered_proba / row_sums

        used_classes = classes_in_model[mask]
        return normalized_proba, used_classes

def get_cart(gen_c, xtrain, ytrain, xval, yval, seed):
    gen_c=clean_data(gen_c)
    gen_c = np.array(gen_c).reshape(-1,1)
    len_train, len_val = len(xtrain), len(xval)
    gen_c_train, gen_c_val = gen_c[:len_train], gen_c[len_train:len_train + len_val]
    enc = MinMaxScaler()
    enc = enc.fit(gen_c_train)
    gen_c_train = enc.transform(gen_c_train)
    gen_c_val = enc.transform(gen_c_val)
    new_train = np.concatenate([xtrain, gen_c_train], axis = -1)
    new_val = np.concatenate([xval, gen_c_val], axis = -1)
    best_val = 0
    for j in range(1, 4):
        clf_CART = DecisionTreeClassifier(max_depth=j, random_state=0)
        clf_CART.fit(new_train, ytrain)

        yval_proba = clf_CART.predict_proba(new_val)

        num_classes = len(np.unique(ytrain))
        classes_in_model = clf_CART.classes_

        if num_classes == 2:
            val_auc = roc_auc_score(yval, yval_proba[:, 1])
        else:
            val_auc = roc_auc_score(yval, yval_proba, average='macro', multi_class='ovr')

        if val_auc > best_val:
            best_val = val_auc
            best_CART = clf_CART
    return best_CART

def gen_prompt(r_list, dt_list, score_list, idx):

    s_l_np = np.array(score_list)
    sorted_idx = np.argsort(s_l_np)[-7:]
    new_r = []
    new_dt = []
    new_s = []
    for i in sorted_idx:
        new_r.append(r_list[i])
        new_dt.append(dt_list[i])
        new_s.append(score_list[i])
    
    text = f"I have some rules to generate x{idx} from x1"
    for i in range(1, idx-1):
        text += f", x{i+1}"
    text += ". We also have corresponding decision tree (CART) to predict 'y' from x1"
    for i in range(1, idx):
        text += f", x{i+1}"
    text += ". The rules are arragned in ascending order based on their scores evaluated with XGBoost classifier, where higher scores indicates better quality."
    text += "\n\n"
    for i in range(len(new_r)):
        text += f"Rule to generate x{idx}:\n{new_r[i]}\n"
        text += f"Decision tree (CART):\n{new_dt[i]}"
        text += "Score evaluated with XGBoost classifier:\n{:.0f}".format(new_s[i]*10000)
        text += "\n\n"
    text += f"Give me a new rule to generate x{idx} that is totally different from the old ones and has a score as high as possible. "
    text += f"Decision trees (including both CART and XGBoost classifier) trained with newly generated x{idx} should be better than the old ones. "
    text += f"Write the rule to generate x{idx} from x1"
    for i in range(1, idx-1):
        text += f", x{i+1}"
    text += f" in square brackets. Variables x1 ~ x{idx} are in [0, 1]. You can use various numpy function. Do not use np.log, np.sqrt, np.arcsin, np.arccos, np.arctan. Do not divide. When divide or using log, use (x+1) term. Think creatively. The new rule must be written with Python grammar.Ensure all brackets are properly matched."    
    text += f" Return the rule only with no explanation."
    return text

def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    def recurse(node, depth):
        indent = "  " * depth
        result = ""
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            result += f"{indent}if {name} > {threshold:.2f}:\n"
            result += recurse(tree_.children_right[node], depth + 1)
            result += f"{indent}else:\n"
            result += recurse(tree_.children_left[node], depth + 1)
        else:
            if tree_.value[node][0][0] > tree_.value[node][0][1]:
                result += f"{indent}y = 0.\n"
            else:
                result += f"{indent}y = 1.\n"
        
        return result

    return recurse(0, 0)

def load_model(model_path, peft_model_path=None):
    return None,None
#     config = AutoConfig.from_pretrained(model_path)

#     quantization_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_use_double_quant=True,
#         bnb_4bit_quant_type="nf4",
#         bnb_4bit_compute_dtype=torch.float16,
#     )

#     model = AutoModelForCausalLM.from_pretrained(
#         model_path,
#         config=config,
#         # attn_implementation="flash_attention_2",
#         quantization_config=quantization_config,
#         torch_dtype=torch.float16,
#         device_map="auto",
#         trust_remote_code=True,
#     )
#     if peft_model_path: model.load_adapter(peft_model_path)
#     tokenizer = AutoTokenizer.from_pretrained(model_path)
#     return model, tokenizer
def clean_llm_output(raw_output: str) -> str:
    if '```' in raw_output or 'python' in raw_output.lower():
        raw_output = raw_output.replace('```python', '')
        raw_output = raw_output.replace('```', '')
    return raw_output.strip().replace('\n', '').replace('\r', '')

import openai
def use_api(prompt, model=None, tokenizer=None, temp=None, iters=1):
    responses = []
    for _ in range(iters):
        try:
            client = openai.OpenAI(
                api_key=global_config.LLM["api_key"],
                base_url=global_config.LLM["base_url"]
            )
            response = client.chat.completions.create(
                model=global_config.LLM["llm_model"],
                messages=[
                    {"role": "system",
                        "content": "You are a helpful Python code assistant.Do not generate any irrelevant information."},
                    {"role": "user", "content": prompt},
                ],
                stream=False,
                temperature=global_config.LLM["code_temp"],
                n=1,
            )
            answer = response.choices[0].message.content
            responses.append(clean_llm_output(answer))

        except Exception as e:
            print("API error")
            import sys
            sys.exit(1)
    return responses

