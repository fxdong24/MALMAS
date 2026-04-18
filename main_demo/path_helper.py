import os
import sys
from sklearn.preprocessing import KBinsDiscretizer
from scipy.stats import gaussian_kde
import pandas as pd
import numpy as np
import json
from datetime import datetime

def get_base_dir(levels_up=1):
    try:
        current_path = os.path.abspath(__file__)
    except NameError:
        current_path = os.path.abspath(os.getcwd())
    for _ in range(levels_up):
        current_path = os.path.dirname(current_path)  
    return current_path

def add_base_to_sys_path(levels_up=1):
    base_path = get_base_dir(levels_up)
    if base_path not in sys.path:
        sys.path.append(base_path)
