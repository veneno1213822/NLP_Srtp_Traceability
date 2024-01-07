# import torch
#
# def check_gpu():
#     if torch.cuda.is_available():
#         device_count = torch.cuda.device_count()
#         current_device = torch.cuda.current_device()
#         device_name = torch.cuda.get_device_name(current_device)
#
#         print(f"Number of available GPUs: {device_count}")
#         print(f"Current GPU index: {current_device}")
#         print(f"Current GPU name: {device_name}")
#     else:
#         print("CUDA is not available. Running on CPU.")
#
# check_gpu()
#
#
# import torch
#
# def set_second_gpu():
#     if torch.cuda.is_available():
#         device_count = torch.cuda.device_count()
#         if device_count >= 2(原版small模型，epoch=1，batch_size=8):
#             target_device = 1  # 0表示第一个GPU，1表示第二个GPU，以此类推
#             torch.cuda.set_device(target_device)
#             print(f"Switched to GPU {target_device}.")
#         else:
#             print("Not enough GPUs available.")
#     else:
#         print("CUDA is not available. Running on CPU.")
#
# set_second_gpu()

import argparse
from transformers import AutoModel, AutoConfig,  AutoTokenizer
from model import MultiSpanQA
import torch
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_path",
    default='./models/chinese_electra_small_L-12_H-256_A-4',  # None,
    type=str,
    # required=True,
    help="Path to Electra model."
)
args = parser.parse_args()

config = AutoConfig.from_pretrained(args.model_path)
# tokenizer = AutoTokenizer.from_pretrained(args.model_path)
pretrain_model = AutoModel.from_pretrained(args.model_path, config=config)
model = MultiSpanQA(pretrain_model)
model1 = torch.load(os.path.join('output', "checkpoint.bin"))
print(model==model1)  # False

# print(hasattr(model, "module"))  # False
# model_to_save = model.module if hasattr(model, "module") else 'model'
# print(model_to_save)


# tokenizer.save_pretrained('./draft')  # 存了三个，special_tokens_map.json和tokenizer_config.json和vocab.txt，这个从一开始就不变