import os
import json
import requests
from flask import Flask, request, jsonify
import subprocess  # torch.cuda.init(); torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
# 输入的格式转换
from convert_internal_external.convert_input_2_model_input import convert_input_2_model_input
# 对象化模型
import oneke_wrapper
from oneke_wrapper import OneKEWrapper
# 后处理：4、文本溯源 detailed
from post_process.text_coref_new import coref
# 后处理：5、加合数据处理
from add_ref.add_refs_new import add_refs_new
# 输出的格式转换
from convert_internal_external.convert_model_output_2_output import convert_model_output_2_output

from train_api import train_app
from Singleton import Singleton_IfTra

# 定义 http_api.py 中的 app 和路由
app = Flask(__name__)
app.register_blueprint(train_app)  # 注册来自 train_api 的 Flask 应用

# 训练和推理不能同时，训练时请求推理会直接return；训练时不能再训练，重叠训练也会直接return
if_training = Singleton_IfTra()

# 中船和华南理工的URL
zhongchuan_url = 'http://127.0.0.1:5000/generate'
huananligong_url = 'http://127.0.0.1:5000/generate'

# oneke模型初始化参数：每次启动都要读取lora_models.json(一直是model_path和lora_path两个项，初始就要有，每次都把微调模型路径修改成最新的，基线模型不变）
# 从lora_models.json读取模型信息
with open('lora_models.json', "r") as file:
    models_info = json.load(file)
# 初始化类属性
model_path = models_info.get("base_model")
lora_model_list = models_info.get("lora_model_list")
# lora_path = lora_model_list[-1]
# model_path = "cache/models--baichuan-inc--Baichuan2-7B-Chat/snapshots/ea66ced17780ca3db39bc9f8aa601d8463db3da5"  # 基座模型
lora_path = "lora/baichuan7B-data2text-continue"  # 微调模型

# 对模型处理方法进行对象化封装。先import对应的py文件，创建对象。后面的模型验证步骤，就是调用改对象的成员方法
oneke_wrapper = OneKEWrapper(model_path, lora_path, "cuda:0", True)
print('推理模型加载完成')

# 更新模型路径：目前是，重新执行./start.sh，重新启动flask
# @app.route('/reinitialize', methods=['POST'])
# def reinitialize_model():
#     # oneke模型初始化参数：每次启动都要读取lora_models.json(一直是model_path和lora_path两个项，初始就要有，每次都把微调模型路径修改成最新的，基线模型不变）
#     # 从lora_models.json读取模型信息
#     with open('lora_models.json', "r") as file:
#         models_info = json.load(file)
#     # 初始化类属性
#     model_path = models_info.get("base_model")
#     lora_model_list = models_info.get("lora_model_list")
#     lora_path = lora_model_list[-1]

# 直接将tasks.json文件的内容加载到 training_tasks 字典中
def read_tasks_history():
    with open('tasks.json', 'r', encoding='utf-8') as file:
        read_tasks = json.load(file)
    return read_tasks

# 模拟推理接口
@app.route('/generate', methods=['POST'])
def inference_data():
    data_dict = request.get_json()  # 接收 JSON 数据

    # 覆盖text部分
    # 遍历 wholeData
    for whole_data_item in data_dict['wholeData']:
        # 遍历 excelData
        for excel_data_item in whole_data_item['excelData']:
            # 用于存储拼接的段落
            combined_text = ""
            # 遍历 segment 中的每一行
            for segment in excel_data_item['segment']:
                # 遍历每一行中的每个列
                for column in segment:
                    # 拼接 columnValue，并在每个值之间添加空格
                    combined_text += column['columnValue'] + ","
            # 用拼接的段落替换原有的 text
            excel_data_item['text'] = combined_text

    # 将处理后的 data_dict 转换回 JSON 字符串
    # processed_data = json.dumps(data_dict, ensure_ascii=False, indent=4)
    print(data_dict)

    # 返回处理后的数据
    return data_dict, 200


# 主流程：接收中电三所发送的数据（无text），转发给中船和华南（现在有text）并接收，进行推理
# 接收中电三所发送的数据的端点：输入param.json（得到的是response.json，不过一开始就输入它也不报错，因为模拟转发时会替换text）
@app.route('/', methods=['POST'])
def receive():
    if if_training.shared_var == True:  # 训练和推理不能同时，训练时请求推理会直接return
        training_tasks = read_tasks_history()
        last_key = list(training_tasks.keys())[-1]; last_value = training_tasks[last_key]
        last_task = {last_key: last_value}
        return jsonify({
            'msg': 'There is currently training in progress, inference is not allowed. Please wait until the training is over.',
            'last_task': last_task
             }), 400

    data = request.json
    with open('./tmp_output/param.json', 'w') as file:
        json.dump(data, file, ensure_ascii=False, indent=2)  # 将字典写入 param.json 文件

    # 1a 将JSON数据转发到中船并接收（对方生成text）
    zhongchuan_response = requests.post(zhongchuan_url, json=data)
    if zhongchuan_response.status_code != 200:
        return jsonify({'error': 'Failed to forward to inference'}), zhongchuan_response.status_code
    res_1 = zhongchuan_response.json()
    with open('./tmp_output/response.json', 'w') as file:
        json.dump(res_1, file, ensure_ascii=False, indent=2)  # 将字典写入 param.json 文件
    print("中船处理后结果:")
    print(res_1)
    # 1b 将JSON数据转发到华南理工并接收（对方生成text）
    huananligong_response = requests.post(huananligong_url, json=data)
    if huananligong_response.status_code != 200:
        return jsonify({'error': 'Failed to forward to inference'}), huananligong_response.status_code
    res_2 = huananligong_response.json()

    # 2 推理结果转化为内部格式 convert_internal_external/convert_input_2_model_input.py
    res_1_i = convert_input_2_model_input(res_1)
    print("推理结果转化为内部格式:")
    print(res_1_i)
    res_2_i = convert_input_2_model_input(res_2)

    # 读出单条
    res_1_i = res_1_i[0]
    res_2_i = res_2_i[0]

    # 3 调用模型处理方法，进行推理验证
    res_1_i_verify = oneke_wrapper.inference_vllm(res_1_i)
    print("调用模型处理方法，进行验证:")
    print(res_1_i_verify)
    res_2_i_verify = oneke_wrapper.inference_vllm(res_2_i)

    # 4 调用溯源方法 post_process/convert_detailed_verify.py
    res_1_i_verify = coref(res_1_i_verify)
    res_2_i_verify = coref(res_2_i_verify)

    # 5 调用加合后处理方法 add_ref/add_refs_new.py
    res_1_i_verify = add_refs_new(res_1_i_verify)
    res_2_i_verify = add_refs_new(res_2_i_verify)

    # 6 结果转化为外部格式 convert_internal_external/convert_model_output_2_output.py
    result = convert_model_output_2_output(res_1_i_verify, res_2_i_verify)

    # 返回预测结果
    return jsonify(result)