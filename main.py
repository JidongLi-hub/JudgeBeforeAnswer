import json
from openai import OpenAI
import requests
from tqdm import tqdm
import os
from utils import *
from model_chat import *
from prompts import Prompts

def pipeline(mllm, llm, image_path, q_type):
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

    # step3 替换前提并生成问题
    generate_question_prompt =  prompt_generater.get_generate_question_prompt(caption, premise)   
    question = llm.chat(generate_question_prompt)

    # step4 生成回答，指出前提错误或者正常回答问题
    answer_prompt = prompt_generater.get_answer_prompt(question, premise)
    answer = mllm.chat(image_path, answer_prompt)

    # step5 产生一条数据
    data = {
        "id":os.path.basename(image_path),
        "image_path":image_path,
        "type":q_type,
        "question":question,
        "label":False,  # False代表前提错误负样本，True代表前提正确正样本
        "premise":premise,
        "answer":answer
    }
    return data



def main():
    image_dir = "/model/fangly/mllm/ljd/dataset/VG_100K/"
    save_file = "./dataset/incorrect_premise_questions.jsonl"
    mllm = MLLM(MLLM_client)
    llm = LLM(LLM_client)
    images = os.listdir(image_dir)
    images = images[40:60]
    q_type = "OCR Content"# "State Attributes" #"Numeric Attributes" #"Visual Attributes" # "Entity Existence"
    # 生成中断恢复
    exists = set()
    try:
        with open(save_file, "r") as f:
            dataset = f.readlines()
        for l in dataset:
            piece = json.loads(l)
            exists.add(piece["id"])
    except:
        pass

    with open(save_file, "a")as f:
        for image in tqdm(images):
            if image in exists:
                continue
            data = pipeline(mllm, llm, os.path.join(image_dir, image), q_type=q_type)
            if data is not None:
                f.write(json.dumps(data) + "\n")
    
    jsonl_to_json(save_file)

if __name__=="__main__":
    main()
