import json
from openai import OpenAI
import requests
from tqdm import tqdm
import os
from utils import *

def VLLM_chat(model_name, client, image_path, text):
    completion = client.chat.completions.create(
    model=model_name,
    messages=[
        {   "role": "user", 
            "content": [{"type": "image_url", 
            "image_url": {"url": "file://"+image_path}},
                {"type": "text", 
            "text":text
            }] 
        }
    ]
    )
    return completion.choices[0].message.content.strip()


def main(model_name ="../models/Qwen2.5-VL-7B-Instruct", port="7001"):
    client = OpenAI(
                    base_url=f"http://localhost:{port}/v1",
                    api_key="00000000",
                )
    test_path = "./dataset/incorrect_premise_questions_Test.json"
    output_path = f'./results/test_results_{model_name.replace("../models/", "")}.jsonl'
    exist_ids = set()
    try:
        with open(output_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                exist_ids.add(json.loads(line)["id"])
    except:
        pass

    with open(test_path, "r") as f:
        data = json.load(f)

    for dic in tqdm(data):
        if dic["id"] in exist_ids:
            continue
        question = dic.get("question", None) 
        image_path = dic.get("image_path")
        if os.path.exists(image_path):             
            try:
                response = VLLM_chat(model_name, client, image_path, question)
            except:
                continue
            result = {
                "id" : dic.get("id", None),
                "image_path": dic.get("image_path", None),
                "type": dic.get("type", None),
                "question":question,
                "label":dic.get("label"),
                "premise":dic.get("premise", None),
                "response":response
            }
            with open(output_path, "a") as f:
                f.write(json.dumps(result) + "\n")
        else:
            print(f"image {image_path} is not exists!")
    jsonl_to_json(output_path)
    print(f"Finished! Stored history to {output_path}")




if __name__=="__main__":
    # main(model_name="../models/Qwen2.5-VL-7B-Instruct", port="7001")
    main(model_name="../models/InternVL3-8B-hf", port="7005")
    # main(model_name="../models/Ola-7b", port="7002")
    # main(model_name="../models/MiniCPM_o_2.6-FlagOS-NVIDIA", port="7003")
    # main(model_name="../models/llava-onevision-qwen2-7b-ov-hf", port="7004")
    
    
    # headers = {"Authorization": "Bearer 00000000"}
    # response = requests.get(url="http://localhost:7777/v1/models", headers=headers)
    # print(response.json())