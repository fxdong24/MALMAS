import os
import sys
from sklearn.preprocessing import KBinsDiscretizer
from scipy.stats import gaussian_kde
import pandas as pd
import numpy as np
import json
from datetime import datetime

def get_base_dir(levels_up=1):
    """
    获取当前文件或当前工作目录向上若干级的路径，兼容 Jupyter 和 .py 脚本。
    
    Args:
        levels_up (int): 向上几级目录（默认是1级）。
    
    Returns:
        str: 上级目录路径的绝对地址。
    """
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
    """
    自动将向上若干级目录添加到 sys.path。
    
    Args:
        levels_up (int): 要添加的路径相对级别。
    """
    base_path = get_base_dir(levels_up)
    if base_path not in sys.path:
        sys.path.append(base_path)

        
        


def enrich_field_info_for_local_pattern(df: pd.DataFrame, field_info: dict, target_col: str = None):
    enriched_info = {}

    for col, info in field_info.items():
        if col not in df.columns:
            continue
        if col == "Id"or col =='PassengerId' or col=='Name':
            enriched_info[col] = info
            continue
        col_data = df[col]
        info = info.copy()
        info["missing_ratio"] = float(col_data.isna().mean())

        # Numerical features
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

            # Try to estimate modality (bimodal/multimodal)
            try:
                kde = gaussian_kde(clean_data)
                grid = np.linspace(clean_data.min(), clean_data.max(), 1000)
                density = kde(grid)
                peaks = np.diff(np.sign(np.diff(density))) < 0
                num_modes = int(peaks.sum())
                info["num_modes_estimated"] = num_modes
            except Exception:
                info["num_modes_estimated"] = None

        # Categorical features
        elif info.get("type") == "categorical":
            value_counts = col_data.value_counts(dropna=False)
            info["n_unique"] = int(col_data.nunique(dropna=True))
            info["values"] = value_counts.index.astype(str).tolist()
            info["value_frequencies"] = value_counts.values.tolist()

        # Optionally add partial dependency or bin-based trend if target is provided
        if target_col and target_col in df.columns:
            try:
                target_data = df[[col, target_col]].dropna()
                if target_data.shape[0] >= 10:
                    est = KBinsDiscretizer(
                        n_bins=10, encode='ordinal', strategy='quantile')
                    bins = est.fit_transform(target_data[[col]])
                    target_data['bin'] = bins.astype(int)
                    bin_means = target_data.groupby(
                        'bin')[target_col].mean().tolist()
                    info["target_bin_avg"] = bin_means
            except Exception:
                info["target_bin_avg"] = None

        enriched_info[col] = info

    return enriched_info