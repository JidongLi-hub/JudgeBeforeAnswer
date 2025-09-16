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

def judge_with_LLM(result_path = "./results/test_results_llava-onevision-qwen2-7b-ov-hf.json"):
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"
    client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )
    models = client.models.list()
    model = models.data[0].id
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
        results = results[:2]
        
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
            response = client.chat.completions.create(
                                model=model, 
                                messages=[
                                        {"role": "user", "content": prompt}
                                    ],
                                extra_body={
                                            "chat_template_kwargs": {"enable_thinking": False},  # 关闭Qwen3的思考模式
                                        }
            )
            print(response.choices[0].message)
            content = response.choices[0].message.content
        except:
            continue
        if "False" in content:
            dic["judge"] = False
        elif "True" in content:
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
    1. 模型成功找出前提错误的问题个数 / 所有问题总数
    2. 其中真正错误的数量 / 模型识别出前提错误的数量
    3. 其中真正正确的数量 / 模型识别出前提正确的数量
    """
    total = len(evaluated_data)

    judged_error_total = sum(1 for d in evaluated_data if d["judge"] is False)
    judged_correct_total = sum(1 for d in evaluated_data if d["judge"] is True)

    true_error_and_judged_error = sum(1 for d in evaluated_data if d["label"] is False and d["judge"] is False)
    true_correct_and_judged_correct = sum(1 for d in evaluated_data if d["label"] is True and d["judge"] is True)

    metric1 = true_error_and_judged_error / total if total > 0 else 0
    metric2 = true_error_and_judged_error / judged_error_total if judged_error_total > 0 else 0
    metric3 = true_correct_and_judged_correct / judged_correct_total if judged_correct_total > 0 else 0

    return metric1, metric2, metric3


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
    for i, name in enumerate(["metric1", "metric2", "metric3"]):
        vals = metrics_samples[i]
        mean = sum(vals) / B
        std = math.sqrt(sum((x - mean) ** 2 for x in vals) / (B - 1))
        half_wide = 1.96 * std / math.sqrt(B)

        # 转换为百分比并保留 1 位小数
        mean_pct = round(mean * 100, 1)
        half_wide_pct = round(half_wide * 100, 1)

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


def main(file_name):
    judge_with_LLM(f"./results/{file_name}")
    with open("evaluate_results/" + file_name) as f:
        evaluated_data = json.load(f)

    # results = bootstrap_metrics(evaluated_data, B=1000, sample_size=5000)
    # save_results_to_csv(results, "evaluate_results/" + file_name.replace(".json", ".csv"))



if __name__=="__main__":
    main(file_name="test_results_Qwen2.5-VL-7B-Instruct.json")
