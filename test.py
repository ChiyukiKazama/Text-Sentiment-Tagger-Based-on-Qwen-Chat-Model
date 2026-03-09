import logging
import os
import torch
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM
import pandas as pd
import time

logging.basicConfig(level=logging.ERROR)

# ====================== 1. 配置项 ======================
MODEL_PATH = ".\models\Qwen-7B-Chat-Int4\qwen\Qwen-7B-Chat-Int4"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(torch.cuda.get_device_name(DEVICE))
MAX_LENGTH = 512
DATASET_SIZE = 100   #从原始数据集中随机抽取进行打标的条数

# ====================== 2. 加载GPTQ量化的Qwen模型  ======================
def load_model():
    """加载本地GPTQ量化的Qwen-7B-Chat-Int4模型"""
    try:
        # 1. 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            local_files_only=True,
            padding_side="right"
        )

        # 确保 pad_token 和 eos_token 正确设置
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if getattr(tokenizer, 'eos_token', None) is None:
            tokenizer.eos_token = '</s>'  # Qwen 默认 eos_token

        # 2. 加载 GPTQ 量化模型（使用 AutoGPTQForCausalLM）
        model = AutoGPTQForCausalLM.from_quantized(
            MODEL_PATH,
            device_map="auto" if torch.cuda.is_available() else DEVICE,
            trust_remote_code=True,
            use_safetensors=True,
            disable_exllama=False,  # 若使用 ExLlama 加速可设为 True（需额外安装）
            inject_fused_attention=True,
            inject_fused_mlp=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,

        )

        model.eval()
        print("✅ 模型加载成功！\n")
        return tokenizer, model

    except Exception as e:
        print(f"❌ 模型加载失败：{str(e)}\n")
        import traceback
        traceback.print_exc()
        raise


# ====================== 3. 加载本地CSV数据集  ======================
def load_dataset(dataset_size, csv_path="./dataset/ChnSentiCorp.csv"):
    """
    从本地CSV文件加载 ChnSentiCorp 酒店评论数据集，并随机抽取 n 条
    """
    if not os.path.exists(csv_path):
        print(f"\n❌ 本地 CSV 文件不存在: {csv_path}")
        return [], []

    try:
        # 使用逗号作为分隔符，并让pandas自动读取第一行为列名
        df = pd.read_csv(csv_path, sep=',')

        # 过滤掉空文本
        df = df.dropna(subset=['review'])

        # 从整个数据集中随机抽取 dataset_size 条记录
        # n=min(dataset_size, len(df)) 确保不会因请求过多而报错
        subset_df = df.sample(n=min(dataset_size, len(df)), random_state=42)


        # 原标签 0->"负面", 1->"正面"
        label_map = {0: "负面", 1: "正面"}
        texts = subset_df['review'].astype(str).tolist()
        labels = [label_map[label] for label in subset_df['label'].tolist()]

        print(f"✅ 成功随机加载 {len(texts)} 条样本。")
        return texts, labels

    except Exception as e:
        print(f"\n❌ CSV 数据集加载失败：{str(e)}")
        return [], []

# ====================== 4. 情感打标核心函数  ======================
def get_sentiment_tag(text, tokenizer, model):
    """
    调用GPTQ量化模型生成中文评论的情感标签
    """
    prompt = (
        "你是一个情感分析助手。请判断以下评论的情感倾向。\n"
        "要求：只输出两个字，只能是“正面”、“负面”之一，不要输出标点符号和其他内容。\n"
        "现在请分析：\n"
        f"评论：{text[:MAX_LENGTH]}\n"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=4,
            do_sample=False,  #GreedySearch
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
            use_cache=True
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    #print(response)

    tag=response[-2:]
    if tag not in["正面","负面"]:
        tag = "未知"

    return tag


# ====================== 5. 主函数  ======================
if __name__ == "__main__":
    print("开始加载数据集...")
    texts, true_labels = load_dataset(DATASET_SIZE)

    print("开始加载模型...")
    tokenizer, model = load_model()

    if not texts:
        print("❌ 未能加载任何数据，程序退出。")
        exit()

    print(f"开始分析 {len(texts)} 条评论...\n")
    print("-" * 60)

    start_time = time.time()
    correct_predictions = 0
    total_predictions = 0

    # 创建一个列表来存储所有结果
    results_data = []

    for i, (text, true_label) in enumerate(zip(texts, true_labels)):
        predicted_label = get_sentiment_tag(text, tokenizer, model)

        if predicted_label == true_label:
            correct_predictions += 1
            status = "✅"
        else:
            status = "❌"

        print(
            f"[{i + 1:3d}/{len(texts):3d}] {status} 真实: {true_label:2s} | 预测: {predicted_label:2s} | 文本: {text[:]}...")

        total_predictions += 1

    end_time = time.time()
    elapsed_time = end_time - start_time

    print("-" * 60)
    accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
    print(f"✅ 分析完成！")
    print(
        f"总计: {total_predictions} 条 | 正确: {correct_predictions} 条 | 错误: {total_predictions - correct_predictions} 条")
    print(f"准确率: {accuracy:.2f}%")
    print(f"总耗时: {elapsed_time:.2f} 秒")
    print(f"平均速度: {total_predictions / elapsed_time:.2f} 条/秒" if elapsed_time > 0 else "平均速度: N/A")
