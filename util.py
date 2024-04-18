import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



def report_cuda_memory():
    t = torch.cuda.get_device_properties(0).total_memory / 1e6
    r = torch.cuda.memory_reserved(0) / 1e6
    a = torch.cuda.memory_allocated(0) / 1e6
    print(f'cuda memory: total = {t}')
    print(f'reserved = {r}')
    print(f'allocated = {a}')
    print(f'free = {r - a}')
    

def output_cuda_memory():
    print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
    print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
    print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))
    
    
def clean_cuda_memory(model = None):
    if model is not None:
        del model
    torch.cuda.empty_cache()
    
    
    
def output_memory_usage(cur_step):
    print(f'memory usage for step {cur_step}:')
    for i in range(torch.cuda.device_count()):
        print(torch.cuda.get_device_properties(i).name)
        print(torch.cuda.memory_allocated(i))
        

def get_major_device(model):
    if hasattr(model, 'device_ids'):
        return f'cuda:{model.device_ids[0]}'
    else:
        return 'cuda:0'
    
def model_total_memory(model):
    mem_params = sum([param.nelement() * param.element_size() for param in model.parameters()])
    mem_bufs = sum([buf.nelement()*buf.element_size() for buf in model.buffers()])
    mem = mem_params + mem_bufs # in bytes
    return mem

def model_nparams(model):
    n_params = sum([param.nelement() for param in model.parameters()])
    return n_params