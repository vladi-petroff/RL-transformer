import numpy as np
import matplotlib.pyplot as plt
import torch


def get_major_device(model):
    if hasattr(model, 'device_ids'):
        return f'cuda:{model.device_ids[0]}'
    else:
        return 'cuda:0'
    

### cleaning up cuda memory 
def clean_cuda_memory(model = None):
    if model is not None:
        del model
    torch.cuda.empty_cache()
    
def output_memory_usage(cur_step):
    print(f'memory usage for step {cur_step}:')
    for i in range(torch.cuda.device_count()):
        print(torch.cuda.get_device_properties(i).name)
        print(torch.cuda.memory_allocated(i))
    

### memory utilization by models
def total_memory_used(model):
    mem_params = sum([param.nelement() * param.element_size() for param in model.parameters()])
    mem_bufs = sum([buf.nelement()*buf.element_size() for buf in model.buffers()])
    total_memory = mem_params + mem_bufs # in bytes
    return total_memory

def model_nparams(model):
    n_params = sum([param.nelement() for param in model.parameters()])
    return n_params
