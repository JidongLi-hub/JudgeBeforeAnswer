import json
from openai import OpenAI
import requests
from tqdm import tqdm
import os

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


def main():
    model_name = "models/InternVL3-8B-hf"  # "models/Qwen2.5-VL-7B-Instruct"
    client = OpenAI(
                    base_url="http://localhost:7778/v1",
                    api_key="00000000",
                )
    test_path = "./attributes_test.json"
    output_path = f'./test_results/test_results_{model_name.replace("models/", "")}.jsonl'
    with open(test_path, "r") as f:
        data = json.load(f)
    # data = data[::34]
    for dic in tqdm(data):
        question = dic.get("q", None)
        qid = dic.get("qid", None)
        image_id = dic["tuplist"][0]["irr_imid"]
        image_path1 = "/model/fangly/mllm/ljd/dataset/VG_100K/" + str(image_id) + ".jpg"
        image_path2 = "/model/fangly/mllm/ljd/dataset/VG_100K_2/" + str(image_id) + ".jpg"
        image_id = "000000000000"+ str(image_id)
        image_id = "COCO_val2014_" + image_id[-12:]  + ".jpg"
        image_path3 = "/model/fangly/mllm/ljd/dataset/train2014/" + image_id
        image_path4 = "/model/fangly/mllm/ljd/dataset/val2014/" + image_id
        image_paths = [image_path1, image_path2, image_path3, image_path4]
        for image_path in image_paths:
            if os.path.exists(image_path):             
                if question is not None:
                    try:
                        response = VLLM_chat(model_name, client, image_path, question)
                    except:
                        continue
                result = {
                    "question":question,
                    "question_id":qid,
                    "image_id":os.path.basename(image_path),
                    "response":response
                }
                with open(output_path, "a") as f:
                    f.write(json.dumps(result) + "\n")
                break
    print(f"Finished! Stored history to {output_path}")




if __name__=="__main__":
    main()
    # headers = {"Authorization": "Bearer 00000000"}
    # response = requests.get(url="http://localhost:7777/v1/models", headers=headers)
    # print(response.json())