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
#print(torch.cuda.get_device_name(DEVICE))
MAX_LENGTH = 512
DATASET_SIZE = 20   # 从原始数据集中随机抽取进行打标的条数

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
            disable_exllama=False,
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
        subset_df = df.sample(n=min(dataset_size, len(df)))


        # 提取评论文本
        texts = subset_df['review'].astype(str).tolist()

        print(f"✅ 成功随机加载 {len(texts)} 条样本。")
        return texts

    except Exception as e:
        print(f"\n❌ CSV 数据集加载失败：{str(e)}")
        return []

# ====================== 4. 情感打标核心函数  ======================
def get_tag(text, tokenizer, model):
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
            do_sample=False,  # GreedySearch
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
    texts = load_dataset(DATASET_SIZE)

    print("开始加载模型...")
    tokenizer, model = load_model()

    if not texts:
        print("❌ 未能加载任何数据，程序退出。")
        exit()

    print(f"开始为 {len(texts)} 条评论进行情感打标...\n")
    print("-" * 60)

    start_time = time.time()
    total_tags = 0

    # 创建一个列表来存储原始评论
    input_data = []
    # 创建一个列表来存储所有结果
    output_data = []

    for i, text in enumerate(texts):
        tag = get_tag(text, tokenizer, model)

        print(
            f"[{i + 1:3d}/{len(texts):3d}] 标签: {tag:2s} | 文本: {text[:]}...")

        total_tags += 1

        # 将每一条预测结果和原文本添加到一个列表中
        output_data.append({'Sentiment': tag, 'Comment': text})
        # 将原始评论添加到另一个列表中
        input_data.append({'Comment': text})

    end_time = time.time()
    elapsed_time = end_time - start_time

    print("-" * 60)
    print(f"✅ 打标完成！")
    print(f"总计处理: {total_tags} 条评论")
    print(f"总耗时: {elapsed_time:.2f} 秒")
    print(f"平均速度: {total_tags / elapsed_time:.2f} 条/秒" if elapsed_time > 0 else "平均速度: N/A")

    time_suffix = str(time.time())[-6:]    # 输入输出csv文件名使用生成时间作为后缀
    # 将原始评论保存到CSV文件
    if input_data:
        input_df = pd.DataFrame(input_data)
        input_filename = f"input_{time_suffix}.csv"
        input_df.to_csv(f'./example/input/{input_filename}', index=False, encoding='utf-8')
        print(f"\n打标结果输出已保存至: ./example/input/{input_filename}")

    # 将打标结果保存到带标签的CSV文件
    if output_data:
        output_df = pd.DataFrame(output_data)
        output_filename = f"output_{time_suffix}.csv"
        output_df.to_csv(f'./example/output/{output_filename}', index=False, encoding='utf-8')
        print(f"打标结果输出已保存至: ./example/output/{output_filename}")

