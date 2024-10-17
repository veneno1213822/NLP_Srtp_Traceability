import os
import json
import torch
from model import MultiSpanQA
from transformers import AutoModel, AutoConfig, AutoTokenizer, BertTokenizer
import uuid
import run_predict
import subprocess

class BertWrapper:
    'Bert模型封装类'
    model_path = "../output/checkpoint-400/"
    base_model_path = "../models/bert-base-chinese"
    device = "cuda:1"

    def __init__(self, model_path,base_model_path,device):

        self.model_path = model_path
        self.base_model_path = base_model_path
        self.device = device

        # 加载模型
        config = AutoConfig.from_pretrained(base_model_path)
        pretrain_model = AutoModel.from_pretrained(base_model_path, config=config)
        model = MultiSpanQA(pretrain_model)
        model.to(device=device)
        model = torch.load(os.path.join(model_path, "checkpoint.bin"))  # 加载模型
        tokenizer = BertTokenizer.from_pretrained(model_path)  # 加载标记器

    def inference(self, external_data_csc, external_data_scut, internal_data_csc, internal_data_scut):

        # 生成唯一路径
        unique_id = str(uuid.uuid1())
        top_dir = f"../Test_requests/Test_Request_{unique_id}"
        os.makedirs(f"{top_dir}/data", exist_ok=True)

        # 保存两家返回的数据
        json.dump(external_data_csc, open(top_dir + "/data/webTestData_aug_1.json", "w+"))
        json.dump(external_data_scut, open(top_dir + "/data/webTestData_aug_2.json", "w+"))
        json.dump(internal_data_csc, open(top_dir + "/data/webTestData_model_1.json", "w+"))
        json.dump(internal_data_scut, open(top_dir + "/data/webTestData_model_2.json", "w+"))

        # 1、preparations.py + 2、run_data_process：数据预处理
        result = subprocess.run([f"./run_preprocess.sh {top_dir}"], capture_output=True, text=True, shell=True)

        # 4、run_predict.py：模型预测
        print("----run_prediction----")
        run_predict.main(self.model, self.tokenizer, top_dir)
        print("----prediction_done----")

        # stringpair.py + 5、cal_pred_coordinate_model.py:添加位置信息
        print("----post processing----")
        result = subprocess.run([f'./run_inference.sh {top_dir}'], capture_output=True, text=True, shell=True)
        print("----post processing done----")

        # 读取模型预测结果
        result_csc = json.load(open(f"{top_dir}/TestOutput/Predictions_model_1.json"))
        result_scut = json.load(open(f"{top_dir}/TestOutput/Predictions_model_2.json"))

        return result_csc,result_scut