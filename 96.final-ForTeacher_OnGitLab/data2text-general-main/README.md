# Guideline
- 项目整体任务分配 / 节点安排 / 问题汇总 / 讨论 --> 详见 Issue
- 非涉密文档/数据管理

## 安装环境依赖

### pytorch
```
pip install torch==2.4.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/test/cu121
# 必须指定numpy版本，否则会出错
pip install numpy==1.24.4
```

### vllm 0.5.4
```
# 安装对应版本vllm
wget https://github.com/vllm-project/vllm/releases/download/v0.5.4/vllm-0.5.4-cp310-cp310-manylinux1_x86_64.whl
pip install vllm-0.5.4-cp310-cp310-manylinux1_x86_64.whl

# 安装 C/C++ 编译器
apt-get install build-essential

# 安装 dev版python
apt install python3.10-dev
```

### 安装剩余依赖
```
pip install accelerate==0.33.0 \
    bitsandbytes==0.43.3 \
    fuzzywuzzy==0.18.0 \
    hanlp==2.1.0b60 \
    huggingface_hub==0.24.6 \
    ngram==4.0.3 \
    peft==0.4.0 \
    Requests==2.32.3 \
    sentencepiece==0.2.0 \
    streamlit==1.38.0 \
    streamlit_chat==0.1.1 \
    word2number==1.1 \
    jsonlines==4.0.0
pip install --ignore-installed Flask==3.0.3
```

## 文件夹介绍
- `internal_data_type`: 存放了内部格式的模型测试数据
- `convert_json、data_inside_type`: 空
- `tmp-test`: 无意义

### internal_data_type文件夹内部文件介绍
格式（参照[[data2text内部数据接口定义]]）

## 加和数据测试示例
 - 人工拆分的加法数据: data2text-test-plus-data-v2.json
 - 模型生成的加法数据: data2text-test-plus-kimi-data-v2.json

## 一般数据测试用例
 - 中电三所提供的一般数据1: data2text-test-normal-data-header1-v2.json
 - 中电三所提供的一般数据2: data2text-test-normal-data-header2-v2.json
 上述两个数据区别在于header不同

## 所有测试数据
 于data2text_test_all-v1.json中
 加和数据：从121400行开始