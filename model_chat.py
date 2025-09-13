import json
from openai import OpenAI
import requests
from tqdm import tqdm
import os

MLLM_client = OpenAI(
                    base_url="http://localhost:7777/v1",
                    api_key="00000000",
                )

LLM_client = OpenAI(
                    base_url="http://localhost:8888/v1",
                    api_key="00000000",
                )
class MLLM:
    def __init__(self, client, model_name="../models/Qwen2.5-VL-72B-Instruct"):
        self.model_name = model_name
        self.client = client
    
    def chat(self, image_path, text):
        completion = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {   "role": "user", 
                                "content": [{"type": "image_url", "image_url": {"url": "file://"+image_path}},
                                            {"type": "text", "text":text
                                }] 
                            }
                        ]
                    )
        return completion.choices[0].message.content.strip()

class LLM:
    def __init__(self, client, model_name="../models/Qwen3-32B"):
        self.client = client
        self.model_name = model_name

    def chat(self, text):
        completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": text}
                ],
                extra_body={
                        "chat_template_kwargs": {"enable_thinking": False},  # 关闭Qwen3的思考模式
                    },
                )
        
        return completion.choices[0].message.content.strip()


    # 发送http请求时要带上Bearer认证
    # headers = {"Authorization": "Bearer 00000000"}
    # response = requests.get(url="http://localhost:7777/v1/models", headers=headers)
    # print(response.json())