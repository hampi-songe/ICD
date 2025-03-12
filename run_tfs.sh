export CUDA_VISIBLE_DEVICES=4
export HF_ENDPOINT=https://hf-mirror.com

# 使用vLLM引擎，不使用AWQ量化（推荐）
python medfound_7b_tfs.py

# # 使用vLLM引擎，使用缩短版系统提示，最大模型长度为16384
# python medfound_7b_test.py --engine vllm --use_short_prompt --max_model_len 16384

# # 使用vLLM引擎，最大模型长度为16384
# python medfound_7b_test.py --engine vllm --max_model_len 16384

# # 使用vLLM引擎，使用缩短版系统提示，使用AWQ量化
# python medfound_7b_test.py --engine vllm --use_short_prompt --use_awq