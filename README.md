# 一种基于大语言模型的文本打标工具

基于LLM（以本地部署的Qwen-7B-Chat-Int4为例），为文本数据自动生成“正面”、“负面“等标签。该项目实现了从环境配置、模型/数据集加载到打标保存的完整流程。

## 快速开始

安装依赖

`pip install -r requirements.txt`

下载Qwen-7B-Chat-Int4用于本地推理

`python download_model.py`

运行实现打标核心逻辑的主程序

`python main.py`

## 项目结构

+ TextSentimentTagger/

  + main.py

  + download_model

  + README.md

  + document.md

  + requirements.txt

  + model/

    + (Qwen-7B-Chat-Int4/)

  + dataset/

    + ChnSentiCorp.csv

  + example/

    + input/
    + output/

    



## 设计与思考

详见document.md







