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