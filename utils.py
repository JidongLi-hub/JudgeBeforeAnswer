import json

def process():
    att_path = "attributes_test.json"
    result_path = "./test_results/test_results_InternVL3-8B-hf.jsonl"
    json_objects = []
    with open(result_path, 'r') as jsonl_file:
        # 逐行读取文件
        for line in jsonl_file:
            # 去除行尾可能存在的空白字符（包括换行符）
            line = line.strip()
            if line:
                # 将每一行的JSON字符串解析为Python对象
                json_objects.append(json.loads(line))
    with open(att_path, "r") as f:
        atts = json.load(f)
    for dic in json_objects:
        for att in atts:
            if att["qid"] == dic["question_id"]:
                dic["attributes"] = att[ "tuplist"][0]
    with open(result_path.replace(".jsonl", ".json"), "w") as f:
        json.dump(json_objects, f, indent=4)
        

def jsonl_to_json(jsonl_file_path, json_file_path):
    """
    将JSONL文件转换为包含JSON对象数组的JSON文件。

    :param jsonl_file_path: 输入的JSONL文件路径。
    :param json_file_path: 输出的JSON文件路径。
    """
    try:
        # 用于存储从JSONL文件中读取的所有对象
        json_objects = []

        with open(jsonl_file_path, 'r') as jsonl_file:
            # 逐行读取文件
            for line in jsonl_file:
                # 去除行尾可能存在的空白字符（包括换行符）
                line = line.strip()
                if line:
                    # 将每一行的JSON字符串解析为Python对象
                    json_objects.append(json.loads(line))

        with open(json_file_path, 'w') as json_file:
            # 将包含所有对象的列表写入到JSON文件中
            # indent=4 用于美化输出，使其更具可读性
            json.dump(json_objects, json_file, ensure_ascii=False, indent=4)
        
        print(f"成功将 '{jsonl_file_path}' 转换为 '{json_file_path}'。")

    except FileNotFoundError:
        print(f"错误：找不到文件 '{jsonl_file_path}'。")
    except json.JSONDecodeError as e:
        print(f"错误：解析JSONL文件时出错。请检查文件格式。错误信息: {e}")
    except Exception as e:
        print(f"发生未知错误: {e}")
