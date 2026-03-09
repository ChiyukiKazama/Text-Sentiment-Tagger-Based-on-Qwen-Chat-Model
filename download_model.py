from modelscope import snapshot_download

model_dir = snapshot_download(
    model_id="qwen/Qwen-7B-Chat-Int4",
    cache_dir="./models/Qwen-7B-Chat-Int4",  # 模型文件路径
    revision="master"
)
print(f"模型已下载到：{model_dir}")