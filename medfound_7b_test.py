import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from peft import PeftModel

# 系统Prompt设计（保持不变）
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


def process_clinical_records(input_path, output_path):
    # 读取输入文件
    with open(input_path, 'r', encoding='utf-8') as f:
        records = json.load(f)

    # 加载本地模型和tokenizer
    model_path = "model/MedFound-7B"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16
    ).eval()

    # # # 加载LoRA适配器
    # model = PeftModel.from_pretrained(model, "./icd_lora_adapter")
    # model = model.merge_and_unload()  # 合并权重方便部署

    results = []

    for record in tqdm(records, desc="Processing Records", unit="case"):
        # 拼接临床文本（保持不变）
        clinical_text = f"""主诉：{record["主诉"]}
现病史：{record["现病史"]}
既往史：{record["既往史"]}
个人史：{record["个人史"]}
入院诊断：{record["入院诊断"]}
诊疗经过：{record["诊疗经过"]}
出院诊断：{record.get("出院诊断", "")}"""

        try:
            # 构建对话格式
            input_text = f"### System:{SYSTEM_PROMPT}\n\n### User:{clinical_text}\n\n### Assistant:"
            input_ids = tokenizer.encode(input_text, return_tensors="pt", add_special_tokens=False).to(model.device)

            # 生成参数设置
            output_ids = model.generate(
                input_ids,
                max_new_tokens=4096,
                temperature=0.3,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

            # 解码输出
            generated_text = tokenizer.decode(output_ids[0, len(input_ids[0]):], skip_special_tokens=True)

            # 清洗结果（保持不变）
            clean_output = generated_text.replace("，", ";").replace("；", ";").replace("｜", "|")

            results.append({
                "病案标识": record["病案标识"],
                "预测结果": clean_output
            })

        except Exception as e:
            print(f"处理病历 {record['病案标识']} 时出错：{str(e)}")
            results.append({
                "病案标识": record["病案标识"],
                "预测结果": "[ERROR]"
            })

    # 保存结果（保持不变）
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    process_clinical_records("tianchi_ICD/ICD-Coding-test-A.json", "tianchi_ICD/pred_result.json")