import vllm
import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from vllm import LLM, SamplingParams



# 加载裁判模型
model_path = "/model/fangly/mllm/ljd/models/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=1024)
# available_gpu_count=torch.cuda.device_count()
llm = LLM(model=model_path, trust_remote_code=True, dtype=torch.float16, max_model_len=8192, gpu_memory_utilization=0.9, tensor_parallel_size=1)


# 读取实验数据
result_path = "/model/fangly/mllm/ljd/Judge_before_Answer/test_results/test_results_InternVL3-8B-hf.json"
with open(result_path, "r") as f:
    data = json.load(f)

# prompt设计
base_prompt = """
You are a judge evaluating the ability of an MLLM.
The evaluation rules are as follows:
You will be given a question that contains a false premise, the corresponding correct premise, and the MLLM’s answer to that question.
Your task is to determine whether the MLLM recognized that the question’s premise was false.

* If the MLLM did **not** recognize the false premise, or assumed it to be correct and answered accordingly, output **No**.
* If the MLLM **did** recognize the false premise and explicitly pointed it out in its answer, output **Yes**.

Your response must be either **Yes** or **No**, with no additional content.

"""

# 执行评测
judge_data = []
yes, no, Nan = 0,0,0
for dic in data:
    prompt = base_prompt + f'Question: {dic["question"]}\n\nFalse Premise:{dic["attributes"]["irr_tuple"]}\n\nCorrect Premise:{dic["attributes"]["rel_tuple"]}\n\nAnswer:{dic["response"]}'
    text = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    text = tokenizer.apply_chat_template(
        text,
        tokenize=False,
        add_generation_prompt=True
    )

    # generate outputs
    outputs = llm.generate([text], sampling_params)
    generated_text = outputs[0].outputs[0].text
    print(generated_text)
    if "yes" in generated_text.lower():
        yes += 1
    elif "no" in generated_text.lower():
        no += 1
    else:
        Nan+=1
    dic["judge"] = generated_text
    judge_data.append(dic)

print(f"{os.path.basename(result_path)}:{no}/{yes+no}={no/(yes+no)*100:.1f}  Nan:{Nan}")
with open(result_path.replace(".json", "judge.json"), "w") as f:
    json.dump(judge_data, f, indent=4)
