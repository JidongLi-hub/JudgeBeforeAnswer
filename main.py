import json
from openai import OpenAI
import requests
from tqdm import tqdm
import os
from utils import *
from model_chat import *
from prompts import Prompts
import random

def pipeline(mllm, llm, image_path, q_type, label=False):
    """ 生成前提错误数据集pipeline """

    prompt_generater = Prompts(q_type)
    # step1 根据前提筛选图片
    judge_prompt = prompt_generater.get_judge_prompt()
    premise = mllm.chat(image_path, judge_prompt)

    # step2 使用筛选结果，生成关于该前提的caption
    if premise.lower() == "no":
        return  # 图片不符合要求
    else:
        caption_prompt = prompt_generater.get_caption_prompt(premise)
        caption = mllm.chat(image_path, caption_prompt)

    # step3 生成问题
    if label:  # 生成前提正确的正样本
        generate_question_prompt =  prompt_generater.get_generate_real_question_prompt(caption, premise)   
        question = llm.chat(generate_question_prompt)
    else:   # 生成替换了前提的错误负样本
        generate_question_prompt =  prompt_generater.get_generate_question_prompt(caption, premise)   
        question = llm.chat(generate_question_prompt)

    # step4 生成回答，指出前提错误或者正常回答问题
    if label:
        answer = mllm.chat(image_path, question)
    else:
        answer_prompt = prompt_generater.get_answer_prompt(question, premise)
        answer = mllm.chat(image_path, answer_prompt)

    # step5 产生一条数据
    data = {
        "id":os.path.basename(image_path),
        "image_path":image_path,
        "type":q_type,
        "question":question,
        "label":label,  # False代表前提错误负样本，True代表前提正确正样本
        "premise":premise,
        "answer":answer
    }
    return data



def main():
    image_dir = "/model/fangly/mllm/ljd/dataset/VG_100K/"
    save_file = "./dataset/incorrect_premise_questions.jsonl"
    type_capacity = 500  # 每种类型问题使用的图片的数量
    mllm = MLLM(MLLM_client)
    llm = LLM(LLM_client)
    q_types = Prompts.supported_types
    images = os.listdir(image_dir)
    images = sample_evenly(images, n=type_capacity*len(q_types)*2)
    nagetive_images = images[::2]
    positive_images = images[1::2]
    exists = set()  # 生成中断恢复
    try:
        with open(save_file, "r") as f:
            dataset = f.readlines()
        for l in dataset:
            piece = json.loads(l)
            exists.add(piece["id"])
    except:
        pass
    
    negative_count, positive_count = {}, {}
    for i, q_type in enumerate(tqdm(q_types)):
        with open(save_file, "a")as f:
            for image in tqdm(nagetive_images[i*type_capacity:(i+1)*type_capacity]):
                if image in exists:
                    continue
                try:
                    nagetive_data = pipeline(mllm, llm, os.path.join(image_dir, image), q_type=q_type, label=False)
                except:
                    print(f"type{q_type}-nagetive image{image} failed-------")
                    continue
                if nagetive_data is not None:
                    f.write(json.dumps(nagetive_data) + "\n")
                    if q_type in negative_count:
                        negative_count[q_type] += 1
                    else:
                        negative_count[q_type] = 1
    for i, q_type in enumerate(tqdm(q_types)):
        with open(save_file, "a")as f:
            for image in tqdm(positive_images[i*type_capacity:(i+1)*type_capacity]):
                if image in exists:
                    continue
                try:
                    positive_data = pipeline(mllm, llm, os.path.join(image_dir, image), q_type=q_type, label=True)
                except:
                    print(f"type{q_type}-positive image{image} failed-------")
                    continue
                if positive_data is not None:
                    f.write(json.dumps(positive_data) + "\n")
                    if q_type in positive_count:
                        positive_count[q_type] += 1
                    else:
                        positive_count[q_type] = 1

    print(f"=========generated {sum(positive_count.values())}positive samples===========")
    print(positive_count)
    print(f"\n\n=========generated {sum(negative_count.values())}nagetive samples===========")
    print(negative_count)

    jsonl_to_json(save_file)

if __name__=="__main__":
    main()
