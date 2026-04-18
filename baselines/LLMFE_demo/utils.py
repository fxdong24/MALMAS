import random
import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype

def is_categorical(x):
    assert type(x) == pd.Series
    x = x.convert_dtypes()
    if is_string_dtype(x):
        return True
  
    elif set(x) == {0, 1}:
        return True
  
    elif x.dtype in [int, float, 'Int64', 'Float64']:
        return False
 
    else:
        return True


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def serialize(row):
    target_str = f""
    for attr_idx, attr_name in enumerate(list(row.index)):
        if attr_idx == 0:
            target_str += "If "
        if attr_idx < len(list(row.index)) - 1:
            target_str += " is ".join([attr_name, str(row[attr_name]).strip(" .'").strip('"').strip()])
            target_str += ", "
        else:
            if len(attr_name.strip()) < 2:
                continue
            target_str += " Then "
            target_str += " is ".join([attr_name, str(row[attr_name]).strip(" .'").strip('"').strip()])
            target_str += "."
    return target_str