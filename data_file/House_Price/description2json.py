
import json
import re

def parse_description_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    results = {}
    current_var = None
    category_pattern = re.compile(r'^\s*([A-Za-z0-9]+)\s+(.*)$')

    for line in lines:
        line = line.strip()

        # 如果是字段定义行，如：MSSubClass: Identifies...
        if ':' in line and not line.startswith((' ', '\t')):
            if current_var:
                results[current_var['name']] = current_var
            name, desc = line.split(':', 1)
            current_var = {
                'name': name.strip(),
                'description': desc.strip(),
                'type': 'numerical',  # 默认连续变量，后续遇到枚举再改为 categorical
            }

        # 如果是分类变量的枚举项（如 20 说明...）
        elif current_var and category_pattern.match(line):
            match = category_pattern.match(line)
            code = match.group(1).strip()
            meaning = match.group(2).strip()

            if current_var['type'] != 'categorical':
                current_var['type'] = 'categorical'
                current_var['values'] = {}

            current_var['values'][code] = meaning


    # 添加最后一个变量
    if current_var:
        results[current_var['name']] = current_var

    return results



# 使用脚本并保存为 JSON 文件
parsed = parse_description_file('data_description.txt')

# 保存为以字段名为键的 JSON
with open('parsed_description.json', 'w', encoding='utf-8') as f:
    json.dump(parsed, f, indent=2, ensure_ascii=False)
