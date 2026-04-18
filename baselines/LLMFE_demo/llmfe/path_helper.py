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
