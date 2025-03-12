import json
import os
import gc
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import infer_auto_device_map
import numpy as np
from typing import List, Dict, Any
from collections import defaultdict
import argparse
import time


# 系统Prompt设计
SYSTEM_PROMPT = """
你是一位资深ICD编码专家，需要根据临床诊疗记录准确生成医保版ICD-10编码。请严格遵守以下规则：

1. 主诊断编码必须且只能从以下候选中选择1个：
(1)I10.x00x032:高血压病3级（极高危）
(2)I20.000:不稳定型心绞痛
(3)I20.800x007:微血管性心绞痛
(4)I21.401:急性非ST段抬高型心肌梗死
(5)I50.900x018:慢性心功能不全急性加重

其他诊断编码必须从以下候选中选择1-4个：
E04.101: 甲状腺结节
E04.102: 甲状腺囊肿
E11.900: 2型糖尿病
E14.900x001: 糖尿病
E72.101: 高同型半胱氨酸血症
E78.500: 高脂血症
E87.600: 低钾血症
I10.x00x023: 高血压病1级（高危）
I10.x00x024: 高血压病1级（极高危）
I10.x00x027: 高血压病2级（高危）
I10.x00x028: 高血压病2级（极高危）
I10.x00x031: 高血压病3级（高危）
I10.x00x032: 高血压病3级（极高危）
I20.000: 不稳定型心绞痛
I25.102: 冠状动脉粥样硬化
I25.103: 冠状动脉粥样硬化性心脏病
I25.200: 陈旧性心肌梗死
I31.800x004: 心包积液
I38.x01: 心脏瓣膜病
I48.x01: 心房颤动
I48.x02: 阵发性心房颤动
I49.100x001: 房性期前收缩[房性早搏]
I49.100x002: 频发性房性期前收缩
I49.300x001: 频发性室性期前收缩
I49.300x002: 室性期前收缩
I49.400x002: 偶发房室性期前收缩
I49.400x003: 频发性期前收缩
I49.900: 心律失常
I50.900x007: 心功能Ⅱ级(NYHA分级) 
I50.900x008: 心功能III级(NYHA分级) 
I50.900x010: 心功能IV级(NYHA分级) 
I50.900x014: KillipII级 
I50.900x015: KillipIII级  
I50.900x016: KillipIV级 
I50.900x018: 慢性心功能不全急性加重
I50.907: 急性心力衰竭
I63.900: 脑梗死
I67.200x011: 脑动脉粥样硬化
I69.300x002: 陈旧性脑梗死
I70.203: 下肢动脉粥样硬化
I70.806: 颈动脉硬化
J18.900: 肺炎
J98.414: 肺部感染
K76.000: 脂肪肝
K76.807: 肝囊肿
N19.x00x002: 肾功能不全
N28.101: 单纯性肾囊肿
Q24.501: 冠状动脉肌桥
R42.x00x004: 头晕
R91.x00x003: 肺诊断性影像异常
Z95.501: 冠状动脉支架植入后状态
Z98.800x612: 乳腺术后

2. 注意要点：
- ICD主诊断编码的判断应遵循以下基本原则：
（1）主诊断应反映患者本次住院的主要医疗需求，通常为对健康危害最大、消耗医疗资源最多或住院时间最长的疾病。
（2）病因诊断优先于临床表现。若住院目的是手术，则手术所针对的疾病为主诊断。
（3）若出院时仍未确诊，应选择最可能的诊断或症状。并发症严重时，可作为主诊断。
- 重点关注电子病历中的"入院诊断"，诊断编码的判断要与电子病历中的"入院诊断"对应的疾病对应
- 其他诊断编码不得超过4个，如果可能的其他诊断编码在4个以上，要输出最可能的4个诊断编码
- 不要添加任何解释性文字
- 确保分号为英文符号
- 必要时结合相关知识进行判断

3. 输出格式必须严格遵循：[主诊断编码|其他诊断编码1;其他诊断编码2;...]
示例：[I21.401|I70.203;I25.103;E78.500]

下面，请你根据下面的临床诊疗记录准确生成ICD-10编码：
"""


def setup_vllm_engine(model_path, tensor_parallel_size=1, max_model_len=4096):
    """
    设置vLLM推理引擎
    
    Args:
        model_path: 模型路径
        tensor_parallel_size: 张量并行大小（多GPU情况下使用）
        max_model_len: 最大模型长度，用于处理长输入
        
    Returns:
        vLLM LLM对象和采样参数
    """

    from vllm import LLM, SamplingParams
    
    # 设置vLLM引擎参数
    engine_kwargs = {
        "model": model_path,
        "tensor_parallel_size": tensor_parallel_size,
        "trust_remote_code": True,
        "max_model_len": max_model_len,  # 增加最大模型长度以支持长输入
        "swap_space": 4,  # 添加交换空间设置
        "enforce_eager": True,  # 强制使用eager模式，避免某些编译问题
    }
            # 初始化vLLM引擎
    llm = LLM(**engine_kwargs)
    
    # 设置默认采样参数
    sampling_params = SamplingParams(
        temperature=0.3,
        max_tokens=4096,
    )
    
    print("vLLM引擎初始化完成")
    return llm, sampling_params
    
def get_shortened_system_prompt():
    """
    获取缩短版的系统提示，用于处理长输入
    """
    return """
你是一位资深ICD编码专家，需要根据临床诊疗记录准确生成医保版ICD-10编码。请严格遵守以下规则：

1. 主诊断编码必须且只能从以下候选中选择1个：
(1)I10.x00x032 (2)I20.000 (3)I20.800x007 (4)I21.401 (5)I50.900x018

其他诊断编码必须从候选列表中选择1-4个。

2. 输出格式必须严格遵循：[主诊断编码|其他诊断编码1;其他诊断编码2;...]
示例：[I21.401|I70.203;I25.103;E78.500]

下面，请你根据下面的临床诊疗记录准确生成ICD-10编码：
"""


def process_clinical_records_vllm(input_path, output_path, model_path="medicalai/MedFound-7B"):
    """
    使用vLLM处理临床记录
    
    Args:
        input_path: 输入JSON文件路径
        output_path: 输出JSON文件路径
        model_path: 模型路径
    """
    # 记录开始时间
    start_time = time.time()

    # 读取输入文件
    with open(input_path, 'r', encoding='utf-8') as f:
        records = json.load(f)
    
    print(f"成功加载 {len(records)} 条记录")
    
    # 初始化vLLM引擎
    print(f"正在使用vLLM加载模型: {model_path}")
    llm, sampling_params = setup_vllm_engine(model_path)
    
    # 准备输入数据
    prompts = []
    record_ids = []
    
    print(f"准备处理 {len(records)} 条记录...")
    
    for record in records:
        # 拼接临床文本
        clinical_text = f"""主诉：{record["主诉"]}
现病史：{record["现病史"]}
既往史：{record["既往史"]}
个人史：{record["个人史"]}
入院诊断：{record["入院诊断"]}
诊疗经过：{record["诊疗经过"]}
出院诊断：{record.get("出院诊断", "")}"""

        # 构建对话格式
        input_text = f"### System:{SYSTEM_PROMPT}\n\n### User:{clinical_text}\n\n### Assistant:"
        
        prompts.append(input_text)
        record_ids.append(record["病案标识"])
    
    # 使用vLLM批量处理
    print("开始使用vLLM进行批量推理...")
    outputs = llm.generate(prompts, sampling_params)
    
    # 处理结果
    results = []
    for output, record_id in zip(outputs, record_ids):
        generated_text = output.outputs[0].text
        clean_output = generated_text.replace("，", ";").replace("；", ";").replace("｜", "|")
        results.append({
            "病案标识": record_id,
            "预测结果": clean_output
        })
    
    # 保存结果
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 计算并显示总用时
    end_time = time.time()
    total_time = end_time - start_time
    avg_time = total_time / len(records) if records else 0
    
    print("\n" + "="*50)
    print("推理完成统计")
    print("="*50)
    print(f"总用时: {total_time:.2f}秒")
    print(f"平均每样本: {avg_time:.2f}秒")
    print(f"样本数量: {len(records)}")
    print(f"吞吐量: {len(records)/total_time:.2f} 样本/秒")
    print(f"结果已保存到: {output_path}")
    print("="*50)
    
    return True


def process_clinical_records(input_path, output_path, batch_size=4, use_4bit=True, compile_model=True):
    """
    批量处理临床记录以加速推理
    
    Args:
        input_path: 输入JSON文件路径
        output_path: 输出JSON文件路径
        batch_size: 批处理大小
        use_4bit: 是否使用4位量化以提高速度
        compile_model: 是否使用torch.compile()预编译模型
    """
    # 记录开始时间
    start_time = time.time()
    
    # 读取输入文件
    with open(input_path, 'r', encoding='utf-8') as f:
        records = json.load(f)
    
    # 设置环境变量以提高性能
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    
    # 检测CUDA是否可用
    if torch.cuda.is_available():
        # 使用更高的CUDA工作流模式
        torch.backends.cudnn.benchmark = True
    
    # 加载模型配置
    model_path = "medicalai/MedFound-7B"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"正在加载模型到设备: {device}")
    
    # 使用优化的模型加载方式
    quantization_config = None
    if use_4bit and device == "cuda":
        from transformers import BitsAndBytesConfig
        # 使用4位量化以提高推理速度
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # 优化模型加载
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.float16,
        "device_map": "auto",
    }
    
    if quantization_config:
        model_kwargs["quantization_config"] = quantization_config
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs).eval()
    
    # 启用flash attention 2（如果可用） - 可大幅提升生成速度
    if device == "cuda" and hasattr(model.config, "attn_implementation"):
        try:
            model.config.attn_implementation = "flash_attention_2"
            print("已启用Flash Attention 2")
        except Exception as e:
            print(f"无法启用Flash Attention 2: {e}")
    
    # 使用torch.compile预编译模型以提高速度（在PyTorch 2.0+上可用）
    if compile_model and hasattr(torch, 'compile') and device == "cuda":
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("已使用torch.compile()预编译模型")
        except Exception as e:
            print(f"无法编译模型: {e}")
    
    # 创建临床文本批次
    batches = []
    current_batch = []
    batch_record_ids = []
    
    for record in records:
        # 拼接临床文本
        clinical_text = f"""主诉：{record["主诉"]}
现病史：{record["现病史"]}
既往史：{record["既往史"]}
个人史：{record["个人史"]}
入院诊断：{record["入院诊断"]}
诊疗经过：{record["诊疗经过"]}
出院诊断：{record.get("出院诊断", "")}"""

        # 构建对话格式
        input_text = f"### System:{SYSTEM_PROMPT}\n\n### User:{clinical_text}\n\n### Assistant:"
        
        current_batch.append(input_text)
        batch_record_ids.append(record["病案标识"])
        
        if len(current_batch) == batch_size:
            batches.append((current_batch.copy(), batch_record_ids.copy()))
            current_batch = []
            batch_record_ids = []
    
    # 处理最后一个可能不满的批次
    if current_batch:
        batches.append((current_batch, batch_record_ids))
    
    results = []
    
    # 批量处理
    for batch_texts, batch_ids in tqdm(batches, desc="Processing Batches", unit="batch"):
        try:
            # 批量编码
            inputs = tokenizer(batch_texts, padding=True, return_tensors="pt", add_special_tokens=False).to(device)
            
            # 优化的生成参数
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=device=="cuda"):
                output_ids = model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=4096,  # 根据任务减少tokens数量
                    temperature=0.3,     # 降低温度以获得更确定的输出
                    do_sample=True,     # 针对这种结构化任务，使用贪婪解码更快
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=True       # 确保使用KV缓存加速
                )
                
            # 批量解码
            for i, (output, record_id) in enumerate(zip(output_ids, batch_ids)):
                # 只解码新生成的部分
                input_length = inputs.input_ids[i].shape[0]
                generated_text = tokenizer.decode(output[input_length:], skip_special_tokens=True)
                
                # 清洗结果（保持不变）
                clean_output = generated_text.replace("，", ";").replace("；", ";").replace("｜", "|")
                
                results.append({
                    "病案标识": record_id,
                    "预测结果": clean_output
                })
                
        except Exception as e:
            print(f"处理批次时出错：{str(e)}")
            # 出错时，为批次中的所有记录添加错误标记
            for record_id in batch_ids:
                results.append({
                    "病案标识": record_id,
                    "预测结果": "[ERROR]"
                })
        
        # 及时清理GPU内存
        if device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
    
    # 保存结果（保持不变）
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 计算并显示总用时
    end_time = time.time()
    total_time = end_time - start_time
    print(f"处理完成！结果已保存到 {output_path}")
    print(f"总用时: {total_time:.2f}秒，平均每样本: {total_time/len(records):.2f}秒")


if __name__ == "__main__":
    # 设置命令行参数
    parser = argparse.ArgumentParser(description="医疗ICD编码推理脚本")
    parser.add_argument("--input", type=str, default="./ICD-Coding-test-A.json", help="输入文件路径")
    parser.add_argument("--output", type=str, default="./pred_result.json", help="输出文件路径")
    parser.add_argument("--model", type=str, default="medicalai/MedFound-7B", help="模型路径")
    
    args = parser.parse_args()
    
    # 打印使用信息
    print("\n" + "="*50)
    print("医疗ICD编码推理脚本")
    print("="*50)
    print(f"模型路径: {args.model}")
    print(f"输入文件: {args.input}")
    print(f"输出文件: {args.output}")
    print("="*50 + "\n")
    
    # 执行推理
    process_clinical_records_vllm(
        args.input,
        args.output,
        model_path=args.model
    )