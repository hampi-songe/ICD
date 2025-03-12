import json
from openai import OpenAI
from tqdm import tqdm

# 系统Prompt设计
SYSTEM_PROMPT = """
你是一位资深ICD编码专家，需要根据临床诊疗记录准确生成ICD-10编码。你需要从临床诊疗记录中分析出对应主诊断编码和其他诊断编码的症状，然后对应到相应的编码。
请严格遵守以下规则：

1. 输出格式必须严格遵循：[主诊断编码|其他诊断编码1;其他诊断编码2;...]
示例：[I21.401|I70.203;I25.103;E78.500]
注意：不要添加任何解释性文字,仅按照上述格式输出!

2. 主诊断编码必须且只能从以下候选中选择1个：
[I10.x00x032, I20.000, I20.800x007, I21.401, I50.900x018]

主诊断编码对应的疾病类型如下所示：
I10.x00x032:高血压病3级（极高危）
I20.000:不稳定型心绞痛
I20.800x007:微血管性心绞痛
I21.401:急性非ST段抬高型心肌梗死
I50.900x018:慢性心功能不全急性加重

其他诊断编码必须从以下候选中选择3-7个：
[E04.101, E04.102, E11.900, E14.900x001, E72.101, E78.500, E87.600, I10.x00x023, 
I10.x00x024, I10.x00x027, I10.x00x028, I10.x00x031, I10.x00x032, I20.000, I25.102, 
I25.103, I25.200, I31.800x004, I38.x01, I48.x01, I48.x02, I49.100x001, I49.100x002,
I49.300x001, I49.300x002, I49.400x002, I49.400x003, I49.900, I50.900x007, I50.900x008,
I50.900x010, I50.900x014, I50.900x015, I50.900x016, I50.900x018, I50.907, I63.900,
I67.200x011, I69.300x002, I70.203, I70.806, J18.900, J98.414, K76.000, K76.807,
N19.x00x002, N28.101, Q24.501, R42.x00x004, R91.x00x003, Z54.000x033, Z95.501, Z98.800x612]

其他诊断编码对应的疾病类型如下所示：
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
Z54.000x033: 青光眼术后
Z95.501: 冠状动脉支架植入后状态
Z98.800x612: 乳腺术后

3. 注意要点：
- 编码必须完全匹配候选项
- 不要添加任何解释性文字
- 确保分号为英文符号
- 主诊断必须反映主要疾病
- 其他诊断应包含并发症、合并症等

4、ICD-10编码分析通用流程如下，请你按照以下流程，一步步思考后给出结论：
主诉定位核心症状：提取主诉关键词，映射至症状编码，辅助定位系统疾病。
现病史交叉验证：结合病程时长、确诊依据（影像/检验）、治疗反应，确定疾病活动性与严重度。
既往史合并编码：慢性病若影响当前治疗或预后，需作为合并症编码；已治愈疾病仅作背景记录。
个人史关联修正：职业暴露、吸烟等危险因素可能需作为附加编码，特别在呼吸/肿瘤疾病中。
诊疗经过分层：主要操作决定医疗资源消耗、并发症单独编码、病理结果修正肿瘤形态学编码（M系列）

下面，请你根据临床诊疗记录进行分析和编码：
"""

def process_clinical_records(input_path, output_path):
    # 读取输入文件
    with open(input_path, 'r', encoding='utf-8') as f:
        records = json.load(f)

    client = OpenAI(
        api_key="sk-ctzfnwemdnefpinmyjyiwjcdxycyvujvhmrnlljrmbrcbghk",
        base_url='https://api.siliconflow.cn/v1',
    )

    results = []

    for record in tqdm(records, desc="Processing Records", unit="case"):
        # 拼接临床文本
        clinical_text = f"""主诉：{record["主诉"]}
现病史：{record["现病史"]}
既往史：{record["既往史"]}
个人史：{record["个人史"]}
入院诊断：{record["入院诊断"]}
诊疗经过：{record["诊疗经过"]}
出院诊断：{record.get("出院诊断", "")}"""

        try:
            # 调用API
            response = client.chat.completions.create(
                model="Qwen/Qwen2.5-7B-Instruct",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": clinical_text}
                ],
                temperature=0.1,
                max_tokens=2048,
                top_p=0.6,
                stream=False
            )

            # 解析结果
            raw_output = response.choices[0].message.content.strip()
            # 清洗结果（确保格式正确）
            clean_output = raw_output.replace("，", ";").replace("；", ";").replace("｜", "|")

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

    # 保存结果
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    process_clinical_records("ICD-Coding-test-A.json", "pred_result.json")



def error_merge(self):
    # 读取pred_result.json文件
    with open('pred_result.json', 'r', encoding='utf-8') as pred_file:
        pred_results = json.load(pred_file)

    # 读取error_pred_result.json文件
    with open('error_pred_result.json', 'r', encoding='utf-8') as error_file:
        error_results = json.load(error_file)

    # 创建一个字典用于快速查找error_results中的记录
    error_dict = {record['病案标识']: record['预测结果'] for record in error_results}

    # 遍历pred_results，并根据病案标识更新预测结果
    for record in pred_results:
        if record['病案标识'] in error_dict:
            record['预测结果'] = error_dict[record['病案标识']]

    # 将更新后的pred_results保存到新的json文件中，或者覆盖原文件
    output_file_path = 'pred_result.json'  # 或者使用 'pred_result.json' 覆盖原文件
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        json.dump(pred_results, output_file, ensure_ascii=False, indent=4)

    print(f"处理完成，更新后的预测结果已保存到{output_file_path}")


def error_process(self):
    # 读取pred_result.json文件
    with open('pred_result.json', 'r', encoding='utf-8') as pred_file:
        pred_results = json.load(pred_file)

    # 读取ICD-Coding-test-A.json文件
    with open('ICD-Coding-test-A.json', 'r', encoding='utf-8') as icd_file:
        icd_data = json.load(icd_file)

    # 创建一个列表用于存储错误预测的病历
    error_icd_list = []

    # 遍历pred_results，找到所有预测结果为"[ERROR]"的病案标识
    for record in pred_results:
        if record['预测结果'] == "[ERROR]":
            # 在icd_data中查找该病案标识对应的病历并添加到error_icd_list
            for patient in icd_data:
                if patient['病案标识'] == record['病案标识']:
                    error_icd_list.append(patient)
                    break

    # 将错误预测的病历写入新的json文件
    with open('error_icd.json', 'w', encoding='utf-8') as error_file:
        json.dump(error_icd_list, error_file, ensure_ascii=False, indent=4)

    print("处理完成，错误预测的病历已保存到error_icd.json")