CUDA_VISIBLE_DEVICES=0 vllm serve ../models/Qwen2.5-VL-7B-Instruct --dtype auto --port 7001 --api-key 00000000 --gpu-memory-utilization 0.9  --allowed-local-media-path /model/fangly/mllm/ljd/dataset

CUDA_VISIBLE_DEVICES=4 vllm serve ../models/Ola-7b --dtype auto --port 7002 --api-key 00000000 --gpu-memory-utilization 0.9  --allowed-local-media-path /model/fangly/mllm/ljd/dataset

CUDA_VISIBLE_DEVICES=5 vllm serve ../models/MiniCPM_o_2.6-FlagOS-NVIDIA --dtype auto --port 7003 --api-key 00000000 --gpu-memory-utilization 0.9  --allowed-local-media-path /model/fangly/mllm/ljd/dataset

CUDA_VISIBLE_DEVICES=6 vllm serve ../models/llava-onevision-qwen2-7b-ov-hf --dtype auto --port 7004 --api-key 00000000 --gpu-memory-utilization 0.9  --allowed-local-media-path /model/fangly/mllm/ljd/dataset

CUDA_VISIBLE_DEVICES=7 vllm serve ../models/InternVL3-8B-hf --dtype auto --port 7005 --api-key 00000000 --gpu-memory-utilization 0.9  --allowed-local-media-path /model/fangly/mllm/ljd/dataset