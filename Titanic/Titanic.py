from .path_and_enrich import enrich_field_info_for_local_pattern, add_base_to_sys_path
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import json
import os
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)
add_base_to_sys_path(3)
import global_config
class Titanic:
    @staticmethod
    def read_data():
        """
        读取数据文件并完成以下步骤：

        1. 加载原始数据文件（默认路径为 "Titanic.csv"）。
        2. 自动识别最后一列为目标变量（target）。
        3. 拆分特征 X 和目标 y。
        4. 根据全局配置参数 global_config 进行训练集和测试集划分。
        5. 将标签 y 合并回训练集和测试集 DataFrame 中，以便后续使用目标变量进行特征增强。
        6. 加载预先解析好的字段描述文件 parsed_description.json。
        7. 使用训练集和目标变量对字段描述进行增强（如数值分布、缺失率、分箱与目标均值等）。

        Returns:
            df_train (pd.DataFrame): 带目标变量的训练集数据。
            df_test (pd.DataFrame): 带目标变量的测试集数据。
            target (str): 目标列的列名（字符串形式）。
            description (dict): 原始字段说明字典，来自 parsed_description.json。
            enrich_description (dict): 增强后的字段说明字典，包含统计信息和与目标相关的特征。
        """

        absolute_path = os.path.dirname(__file__)
        csv_path = os.path.join(absolute_path, "Titanic.csv")
        # 设置随机种子后，每次运行会得到相同结果

        # 读取原始数据
        df = pd.read_csv(csv_path)

        df = make_df_numerically_safe(df)

        # 使用最后一列作为目标变量
        target_col = "Survived"
        target = target_col

        # 拆分特征与标签
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # 拆分训练集和测试集
        df_train, df_test, y_train, y_test = train_test_split(
            X, y, test_size=global_config.data_pre["test_size"], random_state=global_config.data_pre["random_state"]
        )

        # 合并 target 回 df_train 和 df_test，用于后续特征增强
        df_train[target_col] = y_train
        df_test[target_col] = y_test

        description_path = os.path.join(absolute_path, "parsed_description.json")

        # 读取字段描述信息
        with open(description_path, 'r', encoding='utf-8') as f:
            description = json.load(f)
        description=fill_values_from_data(description,df)
        enriched_description_path = os.path.join(absolute_path, "enriched_description.json")

        if os.path.exists(enriched_description_path):
            # 文件存在，直接读取
            with open(enriched_description_path, 'r', encoding='utf-8') as f:
                enrich_description = json.load(f)

        else:
            # 文件不存在，调用生成数据
            # 使用训练集构造字段增强信息（如缺失率、数值分布、目标均值趋势等）
            enrich_description = enrich_field_info_for_local_pattern(
                df_train, description, target_col
            )

            # 保存为 JSON 文件
            with open(enriched_description_path, 'w', encoding='utf-8') as f:
                json.dump(enrich_description, f, ensure_ascii=False, indent=2)



        task_description_path = os.path.join(absolute_path, "taskdescription.txt")

        # 读取 target_description.txt 文件
        with open(task_description_path, "r", encoding="utf-8") as f:
            task_description = f.read()

        return df_train.reset_index(drop=True), df_test.reset_index(drop=True), target, task_description, description, enrich_description
    @staticmethod
    def get_seed_list():
        return  [0, 1, 2]

# def convert_numpy_types(obj):
#     if isinstance(obj, dict):
#         return {k: convert_numpy_types(v) for k, v in obj.items()}
#     elif isinstance(obj, list):
#         return [convert_numpy_types(i) for i in obj]
#     elif isinstance(obj, (np.integer, np.int64, np.int32)):
#         return int(obj)
#     elif isinstance(obj, (np.floating, np.float64, np.float32)):
#         return float(obj)
#     elif isinstance(obj, (np.ndarray,)):
#         return obj.tolist()
#     else:
#         return obj
import json
import pandas as pd

def fill_values_from_data(schema, df):
    """
    根据DataFrame数据自动填充JSON schema中的values字段
    
    参数:
    schema: dict - 原始schema描述
    df: pd.DataFrame - 数据
    
    返回:
    dict - 带有填充values字段的schema
    """
    filled_schema = schema.copy()
    
    for col in df.columns:
        if col in filled_schema:
            col_type = filled_schema[col]["type"]
            
            # 只处理分类和有序变量
            if col_type in ["categorical", "ordinal"]:
                # 获取唯一值并排序
                unique_values = sorted(df[col].dropna().unique())
                
                # 转换为字符串形式(避免JSON序列化问题)
                unique_values = [str(v) for v in unique_values]
                
                filled_schema[col]["values"] = unique_values
    
    return filled_schema




def make_df_numerically_safe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. 把 Categorical -> str
    for col in df.select_dtypes(include='category').columns:
        df[col] = df[col].astype(str)

    # 2. 尝试将所有 object 类型中可转为数字的列转为 float
    for col in df.select_dtypes(include='object').columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except Exception:
            pass  # 无法转换则保留原值

    # 3. 所有 int 强转 float
    for col in df.select_dtypes(include=['int', 'int64', 'int32']).columns:
        df[col] = df[col].astype(float)

    return df