import vllm
import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from vllm import LLM, SamplingParams
from openai import OpenAI
from utils import *
from tqdm import tqdm
import random
import math
import csv
from model_chat import *

def judge_with_LLM(result_path = "./results/test_results_llava-onevision-qwen2-7b-ov-hf.json"):
    # openai_api_key = "EMPTY"
    # openai_api_base = "http://localhost:8000/v1"
    # client = OpenAI(
    #         api_key=openai_api_key,
    #         base_url=openai_api_base,
    #     )
    # models = client.models.list()
    # model = models.data[0].id

    llm = LLM(LLM_client)
    output_path = result_path.replace(".json", ".jsonl").replace("results/", "evaluate_results/")
    exist_ids = set()
    try:
        with open(output_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                exist_ids.add(json.loads(line)["id"])
    except:
        pass

    with open(result_path, "r") as f:
        results = json.load(f)
        
    for dic in tqdm(results):
        if dic["id"] in exist_ids:
            continue
        prompt = f"""
                    You are a strict evaluation judge.  
                    You will be given a question and a response.  
                    The question may contain either a correct premise or a false premise.  
                    Your task is to decide whether the response explicitly recognized the false premise.  

                    - If the response detects and points out the false premise, output exactly: False  
                    - If the response ignores or accepts the premise and directly answers the question, output exactly: True  

                    Important:  
                    - The output must be only a single word: either True or False.  

                    Question: {dic["question"]}
                    Response: {dic["response"]}
                """
        try:
            # response = client.chat.completions.create(
            #                     model=model, 
            #                     messages=[
            #                             {"role": "user", "content": prompt}
            #                         ],
            #                     extra_body={
            #                                 "chat_template_kwargs": {"enable_thinking": False},  # 关闭Qwen3的思考模式
            #                             }
            # )
            # print(response.choices[0].message)
            # content = response.choices[0].message.content
            response = llm.chat(prompt)
        except:
            continue
        if "False" in response:
            dic["judge"] = False
        elif "True" in response:
            dic["judge"] = True
        else:
            print("judge failed!!!")
            continue
        with open(output_path, "a") as f:
            f.write(json.dumps(dic) + "\n")

    jsonl_to_json(output_path)
    print(f"Finished! Stored history to {output_path}")


def compute_metrics(evaluated_data):
    """
    计算三项指标:
    1. False Premise Coverage (FPC)模型成功找出前提错误的问题个数 / 所有问题总数 
    2. False Premise Detection Precision (FPDP)其中真正错误的数量 / 模型识别出前提错误的数量
    3. True Premise Identification Rate (TPIR)其中真正正确的数量 / 模型识别出前提正确的数量
    """
    total = len(evaluated_data)

    judged_error_total = sum(1 for d in evaluated_data if d["judge"] is False)
    judged_correct_total = sum(1 for d in evaluated_data if d["judge"] is True)

    true_error_and_judged_error = sum(1 for d in evaluated_data if d["label"] is False and d["judge"] is False)
    true_correct_and_judged_correct = sum(1 for d in evaluated_data if d["label"] is True and d["judge"] is True)

    FPC = true_error_and_judged_error / total if total > 0 else 0
    FPDP = true_error_and_judged_error / judged_error_total if judged_error_total > 0 else 0
    TPIR = true_correct_and_judged_correct / judged_correct_total if judged_correct_total > 0 else 0

    return FPC, FPDP, TPIR


def bootstrap_metrics(evaluated_data, B=1000, sample_size=5000, seed=42):
    """
    使用 bootstrap 方法计算三项指标的均值和 95% CI 半宽度
    """
    random.seed(seed)

    metrics_samples = [[], [], []]  # 分别存放三项指标的所有轮结果

    for _ in range(B):
        sample = [random.choice(evaluated_data) for _ in range(sample_size)]
        m1, m2, m3 = compute_metrics(sample)
        metrics_samples[0].append(m1)
        metrics_samples[1].append(m2)
        metrics_samples[2].append(m3)

    results = {}
    for i, name in enumerate(["FPC", "FPDP", "TPIR"]):
        vals = metrics_samples[i]
        mean = sum(vals) / B
        std = math.sqrt(sum((x - mean) ** 2 for x in vals) / (B - 1))
        half_wide = 1.96 * std / math.sqrt(B)

        # 转换为百分比并保留 1 位小数
        mean_pct = round(mean * 100, 1)
        half_wide_pct = round(half_wide * 100, 2)

        results[name] = {
            "mean±half_wide": f"{mean_pct}±{half_wide_pct}",
            "mean": mean_pct,
            "half-wide": half_wide_pct
        }

    return results


def save_results_to_csv(results, filename="bootstrap_results.csv"):
    """
    保存结果到 CSV
    """
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "mean±half_wide", "mean", "half-wide"])
        for metric, vals in results.items():
            writer.writerow([metric, vals["mean±half_wide"], vals["mean"], vals["half-wide"]])

def classify_data_by_level(data_list):
    """
    根据type字段将数据分类到三个不同的层级
    
    Args:
        data_list: 包含字典的列表，每个字典必须有'type'字段
    
    Returns:
        tuple: (perceptual_data, cognitive_data, reasoning_data) 三个列表
    """
    
    # 定义三个层级的类型
    perceptual_types = {
        "Entity Existence",
        "Visual Attributes", 
        "Numeric Attributes",
        "State Attributes",
        "OCR Content",
        "Symbol Meaning"
    }
    
    cognitive_types = {
        "Spatial Relation",
        "Interaction Relation", 
        "Possessive Relation",
        "Emotion",
        "Scene"
    }
    
    reasoning_types = {
        "Logical",
        "Commonsense"
    }
    
    # 初始化三个结果列表
    perceptual_data = []
    cognitive_data = []
    reasoning_data = []
    
    # 遍历数据进行分类
    for item in data_list:
        item_type = item.get('type')
        
        if item_type in perceptual_types:
            perceptual_data.append(item)
        elif item_type in cognitive_types:
            cognitive_data.append(item)
        elif item_type in reasoning_types:
            reasoning_data.append(item)
        else:
            print(f"Warning: Unknown type '{item_type}' found in data")
    
    return perceptual_data, cognitive_data, reasoning_data


def bootstrap_metrics_by_category(evaluated_data, B=1000, sample_size=5000, seed=42):
    """
    使用 bootstrap 方法计算四类数据（原始数据+三个分级）的指标
    """
    random.seed(seed)
    
    # 先对数据进行分类
    perceptual_data, cognitive_data, reasoning_data = classify_data_by_level(evaluated_data)
    
    # 创建数据字典，包含四类数据
    data_categories = {
        "All": evaluated_data,
        "Perceptual": perceptual_data,
        "Cognitive": cognitive_data,
        "Reasoning": reasoning_data
    }
    
    # 存储所有类别的结果
    all_results = {}
    
    print("Data statistics:")
    for category, data in data_categories.items():
        print(f"{category}: {len(data)} samples")
    print()
    
    # 对每个类别进行bootstrap计算
    for category, data in data_categories.items():
        if len(data) == 0:
            print(f"Warning: {category} has no data, skipping...")
            continue
            
        # 如果数据量小于sample_size，调整sample_size
        actual_sample_size = min(sample_size, len(data))
        if actual_sample_size < sample_size:
            print(f"Warning: {category} has only {len(data)} samples, using {actual_sample_size} as sample size")
        
        metrics_samples = [[], [], []]  # 分别存放三项指标的所有轮结果
        
        for _ in range(B):
            # 从当前类别的数据中采样
            sample = [random.choice(data) for _ in range(actual_sample_size)]
            m1, m2, m3 = compute_metrics(sample)
            metrics_samples[0].append(m1)
            metrics_samples[1].append(m2)
            metrics_samples[2].append(m3)
        
        # 计算每个指标的统计量
        category_results = {}
        for i, name in enumerate(["FPC", "FPDP", "TPIR"]):
            vals = metrics_samples[i]
            mean = sum(vals) / B
            std = math.sqrt(sum((x - mean) ** 2 for x in vals) / (B - 1))
            half_wide = 1.96 * std / math.sqrt(B)
            
            # 转换为百分比并保留适当小数位
            mean_pct = round(mean * 100, 1)
            half_wide_pct = round(half_wide * 100, 2)
            
            category_results[name] = {
                "mean±half_wide": f"{mean_pct}±{half_wide_pct}",
                "mean": mean_pct,
                "half-wide": half_wide_pct
            }
        
        all_results[category] = category_results
        print(f"{category} metrics calculated successfully")
    
    return all_results


def save_category_results_to_csv(all_results, filename="bootstrap_results_by_category.csv"):
    """
    保存所有类别的结果到 CSV 文件
    """
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        
        # 写入表头
        writer.writerow(["category", "metric", "mean±half_wide", "mean", "half-wide"])
        
        # 写入每个类别的结果
        for category, results in all_results.items():
            for metric, vals in results.items():
                writer.writerow([
                    category, 
                    metric, 
                    vals["mean±half_wide"], 
                    vals["mean"], 
                    vals["half-wide"]
                ])
    
    print(f"Results saved to {filename}")


def main(file_name):
    # judge_with_LLM(f"./results/{file_name}")
    with open("evaluate_results/" + file_name) as f:
        evaluated_data = json.load(f)

    # results = bootstrap_metrics(evaluated_data, B=3000, sample_size=6000)
    # save_results_to_csv(results, "evaluate_results/" + file_name.replace(".json", "_all.csv"))
        
    all_results = bootstrap_metrics_by_category(
        evaluated_data, 
        B=1000, 
        sample_size=5000, 
        seed=42
    )
    
    # 保存结果
    save_category_results_to_csv(all_results, "evaluate_results/" + file_name.replace(".json", ".csv"))



if __name__=="__main__":
    main(file_name="test_results_Qwen2.5-VL-7B-Instruct.json")
    main("test_results_llava-onevision-qwen2-7b-ov-hf.json")
    main("test_results_InternVL3-8B-hf.json")
    main("test_results_llava-1.5-7b-hf.json")
