import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoConfig, AutoModelForCausalLM

# class GenRec(nn.Module):

#     def __init__(self, user_num, item_num, device, args):
        
