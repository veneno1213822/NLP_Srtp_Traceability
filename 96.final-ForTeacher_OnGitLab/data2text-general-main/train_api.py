"""本代码为 有多线程"""
import uuid
from flask import Flask, request, jsonify, Blueprint
import threading
import json
import time
from datetime import datetime, timedelta
import shutil
import torch
import sys
import subprocess

from convert_internal_external.convert_input_2_model_input import convert_input_2_model_input

import os  # os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # 在start.sh里设置过了，这里设置没用
import importlib
import oneke_trainer
# 重新加载已经导入的 Python 模块
importlib.reload(oneke_trainer)
from oneke_trainer import OneKETrainer


"""
training_tasks[task_id]['status'] 可能的结果：Training，Preparing，Success，Error
网址：训练'/train'，查询'/tasks/<str:uuid>'（没有uuid就是获取全部）
"""

# 创建一个 Flask 蓝图，用于将该模块的路由注册到主应用
train_app = Blueprint('train_app', __name__)

# 全局变量：
# a. 历史记录文件的路径
tasks_info_file = 'tasks.json'  # tasks文件由程序自动填写
models_info_file = 'lora_models.json'  # models文件需手动填写基座模型和第一个lora，后续程序自动填写

# b.初始化历史记录文件
# # 如果文件不存在，则创建一个空的 JSON 文件；如果存在，则覆盖为空。每次重启容器，tasks.json都会清空（training_tasks本身就是只保留本次服务的历史），lora_models.json会保留。
# with open(tasks_info_file, 'w') as file:
#     json.dump({}, file)
# 如果文件不存在，创建一个空的 JSON 文件；如果存在则不动。除了第一次启动容器初始化tasks.json为空字典，后面每次重启容器都不动tasks.json
if not os.path.exists(tasks_info_file):
    with open(tasks_info_file, 'w') as file:
        json.dump({}, file)

# c. 设置最小训练时间间隔
trainable_time = 1 # 86400秒为24小时

# 直接将tasks.json文件的内容加载到 training_tasks 字典中
def read_tasks_history():
    with open('tasks.json', 'r', encoding='utf-8') as file:
        read_tasks = json.load(file)
    return read_tasks

# c. 保存训练任务的字典，键为任务ID，值为任务详情。training_tasks = {}
training_tasks = read_tasks_history()  # 直接读取tasks.json作为本次服务的全局变量training_tasks（使用实时文件读/写来实现多进程共享变量）
thread_lock = threading.Lock()  # 用于确保线程安全
# d. 手动的线程池
threads_dic = {}
# e. 训练和推理不能同时，训练时请求推理会直接return；训练时不能再训练，重叠训练也会直接return
from Singleton import Singleton_IfTra
if_training = Singleton_IfTra()

# 保存训练任务的历史记录：只保留最新的100条
def save_tasks_history():   # with thread_lock:
    # 获取所有的键并按时间排序
    sorted_keys = sorted(training_tasks.keys(), key=lambda k: training_tasks[k]['start_time'])
    # 检查是否超过100条记录
    if len(sorted_keys) > 100:
        # 找出需要删除的旧键
        keys_to_delete = sorted_keys[:-100]
        # 删除这些旧键对应的条目
        for key in keys_to_delete:
            del training_tasks[key]
    # 将剩余的记录保存为JSON序列化文件
    with open(tasks_info_file, 'w', encoding='utf-8') as json_file:
        json.dump(training_tasks, json_file, ensure_ascii=False, indent=4)
    print(f"training_tasks最新的100条记录已保存到'{tasks_info_file}'")

# 向models文件内添加lora路径信息：只保留最新的100条
def update_models_info(models_info_file, new_lora_path):
    # 读取现有的模型信息
    with open(models_info_file, 'r', encoding='utf-8') as json_file:
        models_info = json.load(json_file)
    # 向 lora_model_list 中添加新模型路径
    models_info['lora_model_list'].append(new_lora_path)
    # 确保 lora_model_list 不超过 100 条
    if len(models_info['lora_model_list']) > 100:
        models_info['lora_model_list'] = models_info['lora_model_list'][-100:]  # 保留最新的 100 条
    # 将更新后的模型信息保存回 JSON 文件
    with open(models_info_file, 'w', encoding='utf-8') as json_file:
        json.dump(models_info, json_file, ensure_ascii=False, indent=4)
    print(f"models_info最新的100条记录已保存到'{models_info_file}'")

# 清理训练文件保存目录
def clean_directories(path, keep_count):
    # 获取该路径下的所有文件夹
    all_folders = [f for f in os.scandir(path) if f.is_dir()]
    # 如果文件夹数量超过指定数量
    if len(all_folders) > keep_count:
        # 按文件夹的创建时间进行排序（从新到旧）
        all_folders.sort(key=lambda f: f.stat().st_ctime, reverse=True)
        # 保留最新的 keep_count 个文件夹
        folders_to_remove = all_folders[keep_count:]
        # 删除多余的文件夹
        for folder in folders_to_remove:
            print(f"Deleting folder: {folder.path}")
            shutil.rmtree(folder.path)  # 删除整个文件夹树

# 训练主逻辑
def train_main(base_model_path, previous_lora_path, data_internal_path):
    # 此处导入以避免循环导入
    from http_api import oneke_wrapper  # 导入http_api中的oneke_wrapper
    # 加载训练模型前，先卸载推理模型避免OOM
    oneke_wrapper.offload_model()
    oneke_train = OneKETrainer(base_model_path, previous_lora_path, data_internal_path)
    start_time, end_time, max_memory_allocated, gpu_index, lora_output_path = oneke_train.train()
    return start_time, end_time, max_memory_allocated, gpu_index, lora_output_path
# 训练测试用
def train_draft(base_model_path, previous_lora_path, data_internal_path):
    time.sleep(10)  # 等待10s
    start_time = -1; end_time = datetime.now().strftime("%Y-%m-%d,%H:%M:%S"); max_memory_allocated = -99; gpu_index = 101; lora_output_path = "lora/baichuan7B-data2text-continue"
    return start_time, end_time, max_memory_allocated, gpu_index, lora_output_path

# 训练函数
def train_model(task_id, data_internal_path, base_model_path, previous_lora_path):
    # 更新任务状态为“正在训练”   # with thread_lock:
    print(f"training_tasks[task_id] 1 = {training_tasks[task_id]}")
    training_tasks[task_id]['status'] = 'Training'
    training_tasks[task_id]['msg'] = 'The model is training.'
    print(f"training_tasks[task_id] 2 = {training_tasks[task_id]}")
    save_tasks_history()  # 保存任务到历史记录
    if_training.shared_var = True  # 禁止进行推理服务
    try:  # 训练
        start_time_nouse, end_time, max_memory_allocated, gpu_index, lora_output_path = train_main(base_model_path,previous_lora_path,data_internal_path)
        # start_time_nouse, end_time, max_memory_allocated, gpu_index, lora_output_path = train_draft(base_model_path,previous_lora_path,data_internal_path)
        # with thread_lock:
        # 更新 任务状态为“训练完成”、输出路径、结束时间…… 等
        training_tasks[task_id]['status'] = 'Success'
        training_tasks[task_id]['msg'] = 'Training task completed.'
        training_tasks[task_id]['finetune_output'] = lora_output_path  # 本次微调输出
        training_tasks[task_id]['end_time'] = end_time  # 结束时间
        training_tasks[task_id]['max_memory_allocated'] = max_memory_allocated  # 最大显存占用
        training_tasks[task_id]['gpu_index'] = gpu_index  # 使用显卡编号
        print(f"training_tasks[task_id] 3 = {training_tasks[task_id]}")
        # 更新lora_models.json
        update_models_info(models_info_file, lora_output_path)
        # return jsonify(training_tasks[task_id]), 200。线程函数无法直接返回值
    except Exception as e:  # Exception捕获模型训练时所有的异常  # with thread_lock:
        training_tasks[task_id]['status'] = 'Error'
        training_tasks[task_id]['msg'] = f'Training task failed: {str(e)}'  #  没训练成功的不应该给结束时间，否则会影响24h间隔。training_tasks[task_id]['end_time'] = datetime.now().strftime("%Y-%m-%d,%H:%M:%S")  # 结束时间
        print(f"training_tasks[task_id] 4 = {training_tasks[task_id]}")
        # return jsonify(training_tasks[task_id]), 400。线程函数无法直接返回值
    finally:  # 无论是否抛出异常，这里的代码都会被执行。当try块或except块中的return语句被执行时，会暂时保存返回值，然后继续执行finally块中的代码。执行完finally块后，才会将之前保存的返回值返回给调用者。
        # with thread_lock:
        print(f"training_tasks[task_id] 5 = {training_tasks[task_id]}")
        save_tasks_history()  # 保存任务到历史记录
        if_training.shared_var = False  # 允许进行推理服务
        # 清空 CUDA 缓存
        torch.cuda.empty_cache()

        # 只有训练状态为Success时才终止并重启Flask服务
        if training_tasks[task_id]['status'] == 'Success':
            print("Training completed successfully. Shutting down and restarting Flask service...")
            subprocess.Popen(["./start.sh"])  # 启动新进程，运行start.sh脚本
            os._exit(0)  # 终止当前 Flask 服务


# 启动训练任务的路由
@train_app.route('/train', methods=['POST'])
def start_training():
    if if_training.shared_var == True:  # 训练时不能再次训练
        last_key = list(training_tasks.keys())[-1]; last_value = training_tasks[last_key]
        last_task = {last_key: last_value}
        return jsonify({
            'msg': 'There is currently training in progress, train again is not allowed. Please wait until the training is over.',
            'last_task': last_task
             }), 400
    # 1. 生成唯一的任务ID
    task_id = str(uuid.uuid1())
    # 2. 用户发送数据集JSON内容，不是数据集文件路径data_path = request.get_json()。
    data_ori = request.get_json()  # 只需要把外部数据转化为内部数据，然后交给哲韬做训练，自己这里不需要oneke_train.verify和internal_data_to_train
    # 将外部格式转为内部格式
    try:
        data = convert_input_2_model_input(data_ori)
    except Exception as e:  # Exception捕获所有其他异常  # with thread_lock:
        training_tasks[task_id] = {
            'task_id': task_id,
            'status': "Error",
            'msg': f'Terrible format for input data, which cause: {str(e)}',
            'start_time': datetime.now().strftime("%Y-%m-%d,%H:%M:%S")  # 什么时间开始训练，这里必须有否则后续没法排序
        }
        save_tasks_history()  # 保存任务到历史记录
        return jsonify(training_tasks[task_id]), 400
                # 只有训练状态为Success时才终止并重启Flask服务
        if training_tasks[task_id]['status'] == 'Success':
            print("Training completed successfully. Shutting down and restarting Flask service...")
            subprocess.Popen(["./start.sh"])  # 启动新进程，运行start.sh脚本
            os._exit(0)  # 终止当前 Flask 服务

    # 3. 检查是否在24小时内已有训练任务进行
    now_time = time.time()
    for task in training_tasks.values():
        # 假设 task['end_time'] 是 "2024-09-04,12:34:56"
        # 将字符串解析为 struct_time 对象（保险些，time.mktime不接受str）
        print(f"task: {task}")
        if 'end_time' in task:  # 只关注之前训练完成的，没训练成功的没有end_time
            time_struct = time.strptime(task['end_time'], "%Y-%m-%d,%H:%M:%S")
            # 将 struct_time 对象转换为 Unix 时间戳
            end_time = time.mktime(time_struct)
            if now_time - end_time < trainable_time : # 86400
                need_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time + 86400))  # with thread_lock: 主线程不需要线程锁
                training_tasks[task_id] = {
                    'task_id': task_id,
                    'status': "Error",
                    'msg': f"You can only train once within 24 hours. Please train again after {need_time}.",  # f"24小时内只能训练一次, 请于{need_time}以后再训练。"
                    'start_time': datetime.now().strftime("%Y-%m-%d,%H:%M:%S")  # 什么时间开始训练，这里必须有否则后续没法排序
                }
                # 自定义异常类，raise
                save_tasks_history()  # 保存任务到历史记录
                return jsonify(training_tasks[task_id]), 400  # jsonify({'error': }), 400

    # 4. 验证训练数据和训练规则
    if not valid_training_data(data):  # with thread_lock: 主线程不需要线程锁
        training_tasks[task_id] = {
            'task_id': task_id,
            'status': "Error",
            'msg': "There is an issue with the content or format of the training data.",  # '训练数据内容或格式有问题',
            'start_time': datetime.now().strftime("%Y-%m-%d,%H:%M:%S")  # 什么时间开始训练，这里必须有否则后续没法排序
        }
        save_tasks_history()  # 保存任务到历史记录
        return jsonify(training_tasks[task_id]), 400  # 每次都是return training_tasks[task_id]（ # return jsonify({'error': '训练数据有误或不满足训练规则'}), 400
    # 捕获异常的方式，返回异常详细信息，而不是简单的 jsonify({'error': '错误信息'}), 400

    # 5. 都没问题：准备开始本次训练
    start_time = time.strftime("%Y-%m-%d,%H:%M:%S", time.localtime(now_time))  # formatted_time
    top_dir = f"data2text/train_data/{task_id}+{start_time}"
    os.makedirs(top_dir, exist_ok=True)  # exist_ok=True使得存在该路径时不报错，默认会报错
    data_internal_path = f"{top_dir}/train_data_internal.json"  # 要用 UUID、时间 命名

    # 从lora_models.json读取模型信息
    with open(models_info_file, "r") as file:
        models_info = json.load(file)
    # 初始化类属性
    base_model_path = models_info.get("base_model"); lora_model_list = models_info.get("lora_model_list")
    previous_lora_path = lora_model_list[-1]

    # 在任务字典中添加新任务的初始状态。training_tasks用JSON序列化，task_id作为UUID。   # with thread_lock: 主线程不需要线程锁
    training_tasks[task_id] = {
        'task_id': task_id,
        'status': "Preparing",
        'msg': "Training task started.",  # '任务已启动',
        'start_time': start_time,  # 什么时间开始训练
        'base_model': base_model_path,  # 使用什么基础模型
        'data_path': data_internal_path,  # 使用什么数据（内部格式）
        'previous_version': previous_lora_path,  # 基于的上个版本是什么
    }
    save_tasks_history()  # 保存任务到历史记录

    # 训练数据只保留5个文件夹
    clean_directories("data2text/train_data/", 4)
    # 将内部数据保存为JSON文件
    with open(data_internal_path, 'w', encoding='utf-8') as file_tmp:
        json.dump(data, file_tmp, ensure_ascii=False, indent=2)

    # 6. 启动一个新线程进行模型训练
    if task_id in threads_dic:  # 如果当前线程存在，则要先等待线程停止
        threads_dic[task_id].join()

    # 启动新的训练线程  # with thread_lock: 主线程不需要线程锁  # 锁住，使其他线程无法修改共享变量(这里的threads_dic)
    thread = threading.Thread(target=train_model, args=(task_id, data_internal_path, base_model_path, previous_lora_path))
    threads_dic[task_id] = thread
    thread.start()  # 将 thread.start() 放在锁外：①避免阻塞其他操作：将 thread.start() 放在锁外可以减少锁的持有时间，避免阻塞其他线程的访问。如果 thread.start() 有时会涉及到复杂的初始化操作，放在锁内会降低并发性能。②降低锁竞争的可能性：尽可能缩小加锁的范围可以减少锁竞争，提高程序的并发性能。

    # 6. 返回任务ID
    return jsonify(training_tasks[task_id]), 200


# 验证训练数据的函数，实际应用中应实现具体的验证逻辑
def valid_training_data(data):   # with thread_lock:
    try:
        # 这里应实现数据验证的具体逻辑，比如训练条数不够。
        return True
    except:
        return False


# 查询特定任务状态的路由
@train_app.route('/tasks/<string:uuid>', methods=['GET'])
def task_info(uuid):   # with thread_lock:
    # 检查任务ID是否存在
    if uuid in training_tasks:
        # 返回任务的详细信息
        return jsonify(training_tasks[uuid]), 200
    # 如果任务ID不存在，返回错误信息
    return jsonify({'error': '任务ID不存在'}), 404


# 从变量查询当前所有训练任务的路由
@train_app.route('/tasks', methods=['GET'])
def task_list():   # with thread_lock:
    # 返回当前所有任务的状态
    return jsonify(training_tasks), 200


