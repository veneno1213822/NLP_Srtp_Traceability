import os
import random
import time
import numpy as np
import torch
from transformers import (
    AutoTokenizer, 
    AutoModel,
    AutoModelForCausalLM, 
    AutoModelForSeq2SeqLM,
    LlamaForCausalLM,
    LlamaTokenizer,
    Seq2SeqTrainer,
    Trainer,
)

MODEL_DICT = {
    "llama":["llama", "llama2", "llama3", "alpaca", "vicuna", "zhixi"],
    "custom":["custom"],
    "falcon":["falcon"],
    "qwen":["qwen"],
    "qwen2":["qwen2"],
    "baichuan":["baichuan", "baichuan2"],
    "chatglm":["chatglm"],
    "moss": ["moss"],
    "cpm-bee": ["cpm-bee"],
    "openba":["openba"],
    "mistral":["mistral"],
}

LORA_TARGET_MODULES_DICT = {
    "llama":['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
    "custom":['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
    "falcon":['query_key_value', 'dense', 'dense_h_to_4h', 'dense_4h_to_h'],
    "baichuan":['W_pack','o_proj','gate_proj','down_proj','up_proj'],
    "chatglm":['query_key_value', 'dense', 'dense_h_to_4h', 'dense_4h_to_h'],
    "qwen":['c_attn', 'attn.c_proj', 'w1', 'w2', 'mlp.c_proj'],
    "qwen2":['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
    "moss": ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
    "cpm-bee": ["project_q", "project_v"],
    "openba":["q", "kv", 'qkv', 'o', 'fc_in', 'fc_out'],
    "mistral": ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
}


def get_model_tokenizer_trainer(model_name):
    if model_name == 'chatglm':
        return AutoModel, AutoTokenizer, Seq2SeqTrainer
    elif model_name == "openba":
        return AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer
    else:
        return AutoModelForCausalLM, AutoTokenizer, Trainer
    

def get_model_name(model_name):
    model_name = model_name.lower()
    for key, values in MODEL_DICT.items():
        for v in values:
            if v == model_name:
                return key
    return "other"


def seed_torch(seed=42):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def get_format_time():
    current_time = int(time.time())
    localtime = time.localtime(current_time)
    dt = time.strftime('%Y:%m:%d %H:%M:%S', localtime)
    return dt



