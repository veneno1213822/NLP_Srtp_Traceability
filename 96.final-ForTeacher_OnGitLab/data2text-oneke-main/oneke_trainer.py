import json
import os
import time
import numpy as np
import jsonlines
import subprocess
import random
from datetime import datetime
import torch
import sys
sys.path.append("./ie2instruction/")
sys.path.append("./src/")

import ie2instruction.convert_func
import src.finetune

from convert.utils.utils import stable_hash, write_to_json
from convert.processer import get_processer
from convert.converter import get_converter
from convert_func import get_train_data_nooption
from convert_func import get_test_data_nooption

from args.parser import get_train_args
from utils.general_utils import (
    LORA_TARGET_MODULES_DICT, 
    seed_torch, 
    get_model_tokenizer_trainer, 
    get_model_name, 
    get_format_time,
)
from src.finetune import train

# 构建在线推理服务


class OneKETrainer:
    '训练封装'

    def __init__(self, model_path, lora_path, data_path):
        print("Initializing Training...")

        # # 读取config文件
        # with open(config_path, "r") as file:
        #     config = json.load(file)

        # 初始化类属性
        self.model_path = model_path
        self.lora_path = lora_path
        self.data_path = data_path
        self.train_test_split_rate = 0.9

    def verify(self):
        # 校验基座模型
        if not os.path.exists(self.model_path):
            raise ValueError(f"Base model path {self.model_path} does not exist.")

        # 校验lora模型
        if not os.path.exists(self.lora_path):
            raise ValueError(f"Lora path {self.lora_path} does not exist.")

        # 创建lora文件夹
        folder_path = os.getcwd()
        current_date = datetime.now().strftime('%Y%m%d')
        folder_name = f"NER_lora_model_{current_date}"
        current_path = os.path.join(folder_path, "lora", folder_name)
        if not os.path.exists(current_path):
            os.makedirs(current_path)

        # 校验训练数据
        if not os.path.exists(self.data_path):
            raise ValueError(f"Training data {data_path} does not exist.")

        return current_path

    def internal_data_to_train(self):

        def rename_file(old_name, new_name):
            # 确保旧文件存在
            if os.path.exists(old_name):
                # 重命名文件
                os.rename(old_name, new_name)
                print(f"File renamed from {old_name} to {new_name}")
            else:
                print(f"File {old_name} does not exist.")

        # 从 JSON 文件中读取数据
        input_file_path = self.data_path
        with open(input_file_path, 'r', encoding='utf-8') as file:
            input_data = json.load(file)

        train_val_set = []
        test_set = []
        all_set = []

        # 遍历输入数据中的每个data项
        for excel in input_data:
            for sheet in excel:
                entities = []

                # 遍历data中的每一行
                for row, row_value in sheet['table']['data'].items():
                    for value in row_value:
                        header_index = row_value.index(value)
                        entity_type = sheet['table']['header'][header_index]
                        if entity_type not in ["国家", "人物", "任务", "行动", "事件"]:
                            if value != "" and "未提及" not in value and "未明确" not in value and "未提供" not in value:
                                entities.append({
                                    "entity": value,
                                    "entity_type": entity_type})
                                if entity_type in ["装备"]:
                                    if "；" in value:
                                        zb_name = []
                                        zb_number = []
                                        sub_values = value.split('；')
                                        for zhuangbei in sub_values:
                                            zb = zhuangbei.split('，')
                                            zb_name.append(zb[0])
                                            zb_number.append(zb[1])
                                            zb_number_set = list(set(zb_number))
                                        for zb_na in zb_name:
                                            entities.append({
                                                "entity": zb_na,
                                                "entity_type": "装备名称"})
                                        for zb_nu in zb_number_set:
                                            entities.append({
                                                "entity": zb_nu,
                                                "entity_type": "装备数量"})
                                    else:
                                        entities.append({
                                            "entity": value,
                                            "entity_type": "装备名称"})

                # 创建输出格式
                output = {"text": sheet["text"], "entity": entities}
                all_set.append(output)

        # 打乱数组顺序
        np.random.shuffle(all_set)

        # 计算分割点
        split_index = int(len(all_set) * self.train_test_split_rate)
        print(len(all_set))

        # 划分训练集和测试集
        train_val_set = all_set[:split_index]
        test_set = all_set[split_index:]

        # 获取当前工作目录
        folder_path = os.getcwd()
        current_date = datetime.now().strftime('%Y%m%d')
        folder_name = f"data2text_{current_date}"
        current_path = os.path.join(folder_path, "data2text", folder_name)
        if not os.path.exists(current_path):
            os.makedirs(current_path)
        output_file_path = os.path.join(current_path, 'data2text-ner-train-val.json')
        test_file_path = os.path.join(current_path, 'data2text-ner-test.json')

        # 打开输出路径
        with open(output_file_path, 'w', encoding='utf-8') as file1:
            json.dump(train_val_set, file1, ensure_ascii=False, indent=1)
        with open(test_file_path, 'w', encoding='utf-8') as file2:
            json.dump(test_set, file2, ensure_ascii=False, indent=1)

        train_json_file = output_file_path
        train_jsonl_file = os.path.join(current_path, 'data2text-ner-train-val.jsonl')
        test_json_file = test_file_path
        test_jsonl_file = os.path.join(current_path, 'data2text-ner-test.jsonl')

        with open(train_json_file, "r", encoding='utf-8') as file:
            with jsonlines.open(train_jsonl_file, 'w') as writer:
                input_data = json.load(file)
                for data in input_data:
                    writer.write(data)

        with open(test_json_file, "r", encoding='utf-8') as file:
            with jsonlines.open(test_jsonl_file, 'w') as writer:
                input_data = json.load(file)
                for data in input_data:
                    writer.write(data)
                    
        # 将文件转化为jsonl的格式，但保持后缀名仍是json
        train_json_file_1 = os.path.join(current_path, 'data2text-ner-train-val-json.json')
        test_json_file_1 = os.path.join(current_path, 'data2text-ner-test-json.json')

        rename_file(train_jsonl_file, train_json_file_1)
        rename_file(test_jsonl_file, test_json_file_1)

        train_val_transformed_path = os.path.join(current_path, 'data2text-ner-train-val-transformed.json')
        test_transformed_path = os.path.join(current_path, 'data2text-ner-test-transformed.json')

        converter_train = get_converter("NER")("zh", NAN='NAN') # 创建一个converter对象
        processer_class_train = get_processer("NER")  # 创建processer_class对象
        processer_train = processer_class_train.read_from_file(
            processer_class_train, "data2text/schema.json", negative=-1
        ) # 对象processer_class使用read_from_file函数读取schema.json文件

        source_train = train_json_file_1.split('/')[-2]  # 用源路径的最后一个文件夹名作为source
        datas_train = processer_train.read_data(train_json_file_1)
        results_train = get_train_data_nooption(datas_train, processer_train, converter_train, 4, False, "NER", "zh", source_train) # 如果是训练数据转化任务，则将数据转化为训练数据格式
        write_to_json(train_val_transformed_path, results_train)


        # 分离训练集和验证集
        with open(train_val_transformed_path, 'r', encoding='utf-8') as file:
            train_val_data = [json.loads(line) for line in file]

        random.shuffle(train_val_data)  # 随机打乱数据
        train_size = int(len(train_val_data) * 0.8)
        train_data = train_val_data[:train_size]
        val_data = train_val_data[train_size:]

        # 生成训练集验证集路径
        train_transformed_path = os.path.join(current_path, 'data2text-ner-train-transformed.json')
        val_transformed_path = os.path.join(current_path, 'data2text-ner-val-transformed.json')

        # 写入训练集和验证集
        with open(train_transformed_path, 'w', encoding='utf-8') as file:
            for entry in train_data:
                file.write(json.dumps(entry, ensure_ascii=False) + '\n')

        with open(val_transformed_path, 'w', encoding='utf-8') as file:
            for entry in val_data:
                file.write(json.dumps(entry, ensure_ascii=False) + '\n')

        converter_test = get_converter("NER")("zh", NAN='NAN') # 创建一个converter对象
        processer_class_test = get_processer("NER")  # 创建processer_class对象
        processer_test = processer_class_test.read_from_file(
            processer_class_test, "data2text/schema_test.json", negative=-1
        ) # 对象processer_class使用read_from_file函数读取schema.json文件

        source_test = test_json_file_1.split('/')[-2]  # 用源路径的最后一个文件夹名作为source
        datas_test = processer_test.read_data(test_json_file_1)
        results_test = get_test_data_nooption(datas_test, processer_test, 4, "NER", "zh", source_test) # 如果是训练数据转化任务，则将数据转化为训练数据格式, split_num, task, language, source
        write_to_json(test_transformed_path, results_test)

        # 将生成的中转文件删除
        if os.path.exists(output_file_path):
            os.remove(output_file_path)
        if os.path.exists(test_file_path):
            os.remove(test_file_path)
        if os.path.exists(train_json_file_1):
            os.remove(train_json_file_1)
        if os.path.exists(test_json_file_1):
            os.remove(test_json_file_1)
        if os.path.exists(train_val_transformed_path):
            os.remove(train_val_transformed_path)

    def train(self):
        # 验证
        lora_output_path = self.verify()

        # 转换
        self.internal_data_to_train()

        # 训练使用的训练集和验证集路径
        folder_path = os.getcwd()
        current_date = datetime.now().strftime('%Y%m%d')
        folder_name = f"data2text_{current_date}"
        current_path = os.path.join(folder_path, "data2text", folder_name)

        train_file_path = os.path.join(current_path, "data2text-ner-train-transformed.json")
        val_file_path = os.path.join(current_path, "data2text-ner-val-transformed.json")
        
        # model和template信息
        if self.model_path == "cache/models--meta-llama--Meta-Llama-3-8B/snapshots/62bd457b6fe961a42a631306577e622c83876cb6":
            model_name = "llama"
            model_template = "alpaca"
        if self.model_path == "cache/models--baichuan-inc--Baichuan2-7B-Chat/snapshots/ea66ced17780ca3db39bc9f8aa601d8463db3da5":
            model_name = "baichuan"
            model_template = "baichuan2"
        if self.model_path == "cache/models--baichuan-inc--BaiChuan2-13B-Chat/snapshots/c8d877c7ca596d9aeff429d43bff06e288684f45":
            model_name = "baichuan"
            model_template = "baichuan2"

        params = {
            "do_train": True,
            "do_eval": True,
            "model_name_or_path": self.model_path,
            "checkpoint_dir": self.lora_path,
            "stage": "sft",
            "model_name": model_name,
            "template": model_template,
            "train_file": train_file_path,
            "valid_file": val_file_path,
            "output_dir": lora_output_path,
            "per_device_train_batch_size": 2,
            "per_device_eval_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "preprocessing_num_workers": 16,
            "num_train_epochs": 10,
            "learning_rate": 5e-5,
            "max_grad_norm": 0.5,
            "optim": "adamw_torch",
            "max_source_length": 400,
            "cutoff_len": 700,
            "max_target_length": 300,
            "evaluation_strategy": "epoch",
            "save_strategy": "epoch",
            "save_total_limit": 10,
            "lora_r": 64,
            "lora_alpha": 64,
            "lora_dropout": 0.05,
            "bits": 4,
            "bf16": True,
        }
    
        model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(params)
        # model_name映射
        model_args.model_name = get_model_name(model_args.model_name)
    
        # 如果为None则通过model_name加载默认的lora_target_modules, 否则加载传入的 
        if finetuning_args.lora_target_modules:  
            finetuning_args.lora_target_modules = eval(finetuning_args.lora_target_modules)
        else:
            finetuning_args.lora_target_modules = LORA_TARGET_MODULES_DICT[model_args.model_name]
        seed_torch(training_args.seed)
        os.makedirs(training_args.output_dir, exist_ok=True)

        # 记录训练开始时间
        start_time = time.time()
        # 记录训练显卡序号
        gpu_index = os.getenv('CUDA_VISIBLE_DEVICES')
        before_memory_allocated = torch.cuda.memory_allocated()
        # 开始训练
        train(model_args, data_args, training_args, finetuning_args, generating_args)
        after_memory_allocated = torch.cuda.memory_allocated()
        # 计算显存最大占用
        max_memory_allocated = after_memory_allocated - before_memory_allocated
        # 记录训练结束时间
        end_time = time.time()

        start_time_formatted = time.strftime('%Y-%m-%d,%H:%M:%S', time.localtime(start_time))
        end_time_formatted = time.strftime('%Y-%m-%d,%H:%M:%S', time.localtime(end_time))

        return start_time_formatted, end_time_formatted, max_memory_allocated, gpu_index, lora_output_path




