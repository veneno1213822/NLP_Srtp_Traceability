import json
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
import gc
from typing import List, Optional, Tuple
from vllm import EngineArgs, LLMEngine, RequestOutput
import os


class OneKEWrapper:
    'OneKE模型封装类'
    model_path = "cache/models--baichuan-inc--Baichuan2-7B-Chat/snapshots/ea66ced17780ca3db39bc9f8aa601d8463db3da5"
    lora_path = "lora/baichuan7B-data2text-continue"
    gpu_mem_before_load = 0
    gpu_mem_after_load = 0
    gpu_mem_max = 0
    model = {}
    tokenizer = {}

    def initialize_engine(self, model: str, quantization: str,
                          lora_repo: Optional[str]) -> LLMEngine:
        """Initialize the LLMEngine."""

        if quantization == "bitsandbytes":
            # QLoRA (https://arxiv.org/abs/2305.14314) is a quantization technique.
            # It quantizes the model when loading, with some config info from the
            # LoRA adapter repo. So need to set the parameter of load_format and
            # qlora_adapter_name_or_path as below.
            engine_args = EngineArgs(
                model=model,
                quantization=quantization,
                qlora_adapter_name_or_path=lora_repo,
                load_format="bitsandbytes",
                enable_lora=True,
                max_lora_rank=64,
                # set it only in GPUs of limited memory
                enforce_eager=True,
                trust_remote_code=True, device=self.device)
        else:
            engine_args = EngineArgs(
                model=model,
                quantization=quantization,
                enable_lora=True,
                max_loras=4,
                # set it only in GPUs of limited memory
                enforce_eager=True, device=self.device)
        engine = LLMEngine.from_engine_args(engine_args)
        return engine

    def process_requests(self, engine: LLMEngine,
                         test_prompts: Tuple[list[int], SamplingParams,
                                                  Optional[LoRARequest]]):
        """Continuously process a list of prompts and handle the outputs."""
        request_id = 0

        if test_prompts:
            prompt, sampling_params, lora_request = test_prompts
            engine.add_request(str(request_id),
                               prompt,
                               sampling_params,
                               lora_request=lora_request)
            request_id += 1

            request_outputs = engine.step()
            # print(request_outputs)
            request_outputs = request_outputs[0]

            print("----------------------------------------------------")
            # print(f"Prompt: {request_outputs.prompt}")
            print(f"Output: {request_outputs.outputs[0].text}")
            print(len(request_outputs.outputs[0].text))

            return request_outputs.outputs[0].text

    def __init__(self, model_path, lora_path, device, vllm=False, quantization=False):
        print("Initializing OneKEWrapper...")
        self.model_path = model_path
        self.lora_path = lora_path
        self.device = device
        self.total_time = 0
        self.data_count = 0
        os.environ['CUDA_VISIBLE_DEVICES'] = device[-1]

        #logger记录
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
        if vllm is False:
            if quantization is False:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    config=config,
                    device_map=device,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                )
            elif quantization == "int8":
                bnb_config = BitsAndBytesConfig(load_in_8bit=True)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    config=config,
                    device_map=device,
                    trust_remote_code=True,
                    quantization_config=bnb_config,
                )
            elif quantization == "int4":
                bnb_config = BitsAndBytesConfig(load_in_4bit=True)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    config=config,
                    device_map=device,
                    trust_remote_code=True,
                    quantization_config=bnb_config,
                )
            self.model = PeftModel.from_pretrained(
                self.model,
                lora_path,
            )
            self.model.eval()
        else:
            # vllm加载模型
            if quantization == "int8":
                config = {
                    'model': self.model_path,
                    'quantization': "bitsandbytes",
                    'lora_repo': self.lora_path
                }
                print(config['model'] + "\n" + config['quantization'] + "\n" + config['lora_repo'])
                self.engine = self.initialize_engine(self.model_path, "bitsandbytes", self.lora_path)
        # 加载模型之后的显存占用
        self.gpu_mem_after_load = torch.cuda.memory_allocated()

    def inference(self, data):
        model = self.model
        tokenizer = self.tokenizer

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
       
        # 更改读入数据
        system_prompt = '<<SYS>>\n 你是一个乐于助人的助手。\n<</SYS>>\n\n'
        input_data = {
            "instruction": "你是一个命名实体识别专家。请从input中抽取符合schema描述的实体，如果实体类型不存在就返回空列表，并输出为可解析的json格式。",
            "schema": data[0]['table']['header'],
            "input": data[0]['text']
        }
        sintruct = json.dumps(input_data, ensure_ascii=False)
        sintruct = '<reserved_106>' + system_prompt + sintruct + '<reserved_107>'
        if self.model_path == "cache/models--meta-llama--Meta-Llama-3-8B/snapshots/62bd457b6fe961a42a631306577e622c83876cb6":
            sintruct = '[INST] ' + system_prompt + sintruct + ' [/INST]'

        input_ids = tokenizer.encode(sintruct, return_tensors="pt").to(device)
        input_length = input_ids.size(1)

        # 记录推理开始时间
        start_time = time.time()

        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=GenerationConfig(
                    max_length=1024,
                    max_new_tokens=256,
                    return_dict_in_generate=True),
                pad_token_id=tokenizer.eos_token_id)

        generation_output = generation_output.sequences[0]
        generation_output = generation_output[input_length:]
        output = tokenizer.decode(generation_output, skip_special_tokens=True)
        # output = re.sub("^\\[", "{",  output)
        # print(output)

        # 记录推理结束时间
        end_time = time.time()
        inference_time = end_time - start_time
        self.logger.info(f"Inference time: {inference_time:.2f} seconds")
        
        result = json.loads(output)
        output_data = add_verify_data(data, result)
        output_data_new = update_verify_based_on_text(output_data)

        # 推理之后的显存占用
        self.gpu_mem_after_load = torch.cuda.memory_allocated()

        self.clear_gpu_cache()
        # # 先写好等实际使用时看情况加
        # # 用logger记录文件，记录推理时间，输入文件大小，推理占用显存
        # # logger记录独特信息
                
        return output_data_new
        
    def inference_vllm(self, data):
        start_time = time.time()
        tokenizer = self.tokenizer
        #设置设备
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
        if "llama" in self.model_path:
            sintruct = '[INST] ' + system_prompt + sintruct + ' [/INST]'
            
        input_ids = tokenizer.encode(sintruct, return_tensors="pt").to(self.device)
        # test_prompts = create_test_prompts(lora_path)
        prompts = (input_ids.tolist(),
                 SamplingParams(temperature=0,
                # logprobs=1,
                # prompt_logprobs=1,
                max_tokens=1024),
             LoRARequest("adapter", 1, self.lora_path))
        output_text = self.process_requests(self.engine, prompts)

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

        self.gpu_mem_after_load = torch.cuda.memory_allocated()

        self.clear_gpu_cache()
        del self.engine
        gc.collect()
        torch.cuda.empty_cache()

    def clear_gpu_cache(self):
        memory_used = self.gpu_mem_after_load - self.gpu_mem_before_load
        memory_occupied = self.gpu_mem_after_load / self.gpu_mem_max * 100

        # 记录到日志
        # self.logger.info(f"Memory used for inference: {memory_used / (1024 ** 2):.2f} MB")
        # self.logger.info(f"Memory occupied after inference: {memory_occupied:.2f}%")



