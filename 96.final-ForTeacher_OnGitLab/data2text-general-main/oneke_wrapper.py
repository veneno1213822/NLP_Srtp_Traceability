import json
import gc
import torch
import time
import logging
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    BitsAndBytesConfig
)
from peft import PeftModel
from verify_oneke.oneke_create_verify import update_verify_based_on_text, add_verify_data
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
# from vllm.model_executor.layers.quantization.bitsandbytes import BitsAndBytesConfig
# from bitsandbytes import BitsAndBytesConfig


class OneKEWrapper:
    'OneKE模型封装类'
    model_path = "cache/models--baichuan-inc--Baichuan2-7B-Chat/snapshots/ea66ced17780ca3db39bc9f8aa601d8463db3da5"
    lora_path = "lora/baichuan7B-data2text-continue"
    gpu_mem_before_load = 0
    gpu_mem_after_load = 0
    gpu_mem_max = 0
    model = {}
    tokenizer = {}

    def __init__(self, model_path, lora_path, device, vllm=False):
        print("Initializing OneKEWrapper...")
        self.model_path = model_path
        self.lora_path = lora_path
        self.device = device
        self.total_time = 0
        self.data_count = 0
        self.vllm = vllm

        # logger记录
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # 最大显存
        device_count = int(self.device[-1])

        self.gpu_mem_max = torch.cuda.get_device_properties(device_count).total_memory

        # # 加载模型之前的显存占用
        self.gpu_mem_before_load = torch.cuda.memory_allocated()
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        # 加载普通模型
        if not vllm:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                config=config,
                device_map=device,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            ).to(device)

            self.model = PeftModel.from_pretrained(
                self.model,
                lora_path,
            )
            # self.model.eval()
        # vllm加载模型
        else:
            print(device)
            self.llm = LLM(model_path,
                           trust_remote_code=True,
                           enable_lora=True,
                           max_lora_rank=64,
                           disable_log_stats=True,
                           gpu_memory_utilization=0.8)
        # 加载模型之后的显存占用
        self.gpu_mem_after_load = torch.cuda.memory_allocated()

    def offload_model(self):
        if self.vllm:
            del self.llm
        else:
            del self.model
        gc.collect()
        torch.cuda.empty_cache()

    def inference(self, data):
        model = self.model
        model.eval()
        tokenizer = self.tokenizer

        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 更改读入数据
        system_prompt = '<<SYS>>\n 你是一个乐于助人的助手。\n<</SYS>>\n\n'
        input_data = {
            "instruction": "你是一个命名实体识别专家。请从input中抽取符合schema描述的实体，如果实体类型不存在就返回空列表，并输出为可解析的json格式。",
            "schema": data[0]['table']['header'],
            "input": data[0]['text']
        }
        sintruct = json.dumps(input_data, ensure_ascii=False)
        sintruct = '<reserved_106>' + system_prompt + sintruct + '<reserved_107>'
        if "llama" in self.model_path:
            sintruct = '[INST] ' + system_prompt + sintruct + ' [/INST]'

        start_time = time.time()

        input_ids = tokenizer.encode(sintruct, return_tensors="pt").to(model.device)
        input_length = input_ids.size(1)

        # 记录推理开始时间

        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=GenerationConfig(
                    max_length=1024,
                    max_new_tokens=512,
                    return_dict_in_generate=True),
                pad_token_id=tokenizer.eos_token_id)

        generation_output = generation_output.sequences[0]
        generation_output = generation_output[input_length:]
        output = tokenizer.decode(generation_output, skip_special_tokens=True)
        # output = re.sub("^\\[", "{",  output)
        print("--------------check output---------------")
        print(output)

        # 记录推理结束时间
        end_time = time.time()
        inference_time = end_time - start_time
        # self.logger.info(f"Inference time: {inference_time:.2f} seconds")

        try:
            result = json.loads(output)
            self.total_time += inference_time
            self.data_count += 1
            output_data = add_verify_data(data, result)
            output_data_new = update_verify_based_on_text(output_data)
            return output_data_new
        except Exception as e:
            print(e)
            print(f"input:{input_data}, output:{output}")
            return output

        # 推理之后的显存占用
        self.gpu_mem_after_load = torch.cuda.memory_allocated()

        self.clear_gpu_cache()
        # # 先写好等实际使用时看情况加
        # # 用logger记录文件，记录推理时间，输入文件大小，推理占用显存
        # # logger记录独特信息

        return output_data_new

    def inference_vllm(self, data):

        tokenizer = self.tokenizer
        # 设置设备
        # device = torch.device(self.device if torch.cuda.is_available() else "cpu")

        # 更改读入数据
        system_prompt = '<<SYS>>\n 你是一个乐于助人的助手。\n<</SYS>>\n\n'
        input_data = {
            "instruction": "你是一个命名实体识别专家。请从input中抽取符合schema描述的实体，如果实体类型不存在就返回空列表，并输出为可解析的json格式。",
            "schema": data[0]['table']['header'],
            "input": data[0]['text']
        }
        sintruct = json.dumps(input_data, ensure_ascii=False)
        sintruct = '<reserved_106>' + system_prompt + sintruct + '<reserved_107>'
        if self.model_path == "./cache/models--meta-llama--Meta-Llama-3-8B/snapshots/62bd457b6fe961a42a631306577e622c83876cb6":
            sintruct = '[INST] ' + system_prompt + sintruct + ' [/INST]'
        # 记录推理开始时间
        start_time = time.time()

        input_ids = tokenizer.encode(sintruct, return_tensors="pt").to(self.device)
        # vLLM 推理
        sampling_params = SamplingParams(
            max_tokens=512,
            temperature=0
        )

        outputs = self.llm.generate(
            sampling_params=sampling_params,
            prompt_token_ids=input_ids.tolist(),  # 转为列表
            use_tqdm=False,
            lora_request=LoRARequest("adapter", 1, self.lora_path),
        )

        output_text = outputs[0].outputs[0].text  # 提取生成的文本
        # self.logger.info(f"Response: {output_text}")

        # 记录推理结束时间
        end_time = time.time()
        inference_time = end_time - start_time

        # self.logger.info(f"Inference time: {inference_time:.2f} seconds")
        try:
            result = json.loads(output_text)
            self.total_time += inference_time
            self.data_count += 1
            output_data = add_verify_data(data, result)
            output_data_new = update_verify_based_on_text(output_data)
            return output_data_new
        except Exception as e:
            print(e)
            print(f"input:{input_data}, output:{output_text}")
            return output_text
        # else:
        self.gpu_mem_after_load = torch.cuda.memory_allocated()

        self.clear_gpu_cache()

    def clear_gpu_cache(self):
        memory_used = self.gpu_mem_after_load - self.gpu_mem_before_load
        memory_occupied = self.gpu_mem_after_load / self.gpu_mem_max * 100

        # 记录到日志
        # self.logger.info(f"Memory used for inference: {memory_used / (1024 ** 2):.2f} MB")
        # self.logger.info(f"Memory occupied after inference: {memory_occupied:.2f}%")



