import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os




import numpy as np
import random
import sys
import csv
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns


from tqdm import tqdm
import time
import sys
import wandb
import argparse
import yaml


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.distributions.categorical import Categorical



from transformers import GPT2Model, GPT2Config
from torch.optim.lr_scheduler import CosineAnnealingLR


from env_MAB import *
from my_algorithms import * 
from util import *
from transformer import *
from train_utils import *





def get_rollouts(model, args):
    
    major_device = get_major_device(model)
    mini_b_size = args['mini_b_size']
    n_arms, dim_size = model.n_arms, 1
    
    mus = sample_mus(mini_b_size, n_arms, regime = args['mus_regime'])
    mus = torch.tensor(mus).cuda()
    max_mus = torch.max(mus, dim = 1, keepdim = True)[0]
    
    def pull_arms(chosen_actions):    
        random_draws = torch.rand((mini_b_size, 1), dtype=float, device='cuda')
        chosen_mus = torch.gather(mus, dim = 1, index = chosen_actions)
        rewards = (random_draws < chosen_mus).float()
        return rewards - 0.5
    
    
    first_actions = torch.tensor(np.random.choice(n_arms, mini_b_size)).view(mini_b_size, -1).cuda() #.to(major_device)
    first_rewards = pull_arms(first_actions).view(mini_b_size, -1).cuda() #.to(major_device)
    
    act_hist = first_actions.view(mini_b_size, -1).float() #.to(major_device)
    rew_hist = first_rewards.view(mini_b_size, -1).float() #.to(major_device)
    model_regrets = torch.tensor([]).view(mini_b_size, 0).float().cuda()
    discounted_rewards = first_rewards.view(mini_b_size, -1).float().cuda()
    discount = 1
    
    past = None
    
    if args['print_memory_usage']:
        output_memory_usage(0)
        
    
    for seq_len in range(args['horizon']):
        discount *= args['gamma']

        if (past is not None) and (seq_len < args['horizon'] - 1):
            output_dict = model(
                act_hist.view(mini_b_size, seq_len + 1, dim_size)[:, -1:, :],
                rew_hist.view(mini_b_size, seq_len + 1, dim_size)[:, -1:, :],
                past = past,
            )
        else:                
            output_dict = model(
                act_hist.view(mini_b_size, seq_len + 1, dim_size),
                rew_hist.view(mini_b_size, seq_len + 1, dim_size),
            )
        
        policy_output, value_output = output_dict['policy'], output_dict['value']
        past = output_dict['past_key_values']
        assert (model.use_cache is False) or (past is not None)

        action_distribution = Categorical(logits = policy_output[:, -1, :].detach())
        new_actions = action_distribution.sample().detach().view(mini_b_size, -1)
        new_rewards = pull_arms(new_actions)

        act_hist = torch.cat((act_hist, new_actions), 1).detach() #.to(major_device)
        rew_hist = torch.cat((rew_hist, new_rewards), 1).detach() #.to(major_device)
        discounted_rewards = torch.cat((discounted_rewards, discount * new_rewards), 1).detach()


        new_model_regrets = discount * (max_mus - torch.gather(mus, dim = 1, index = new_actions))
        model_regrets = torch.cat((model_regrets, new_model_regrets), 1)


        del new_actions, new_rewards, new_model_regrets, action_distribution, output_dict
        if args['print_memory_usage']:
            output_memory_usage(seq_len + 1)
            
                
    return {'act_hist': act_hist, 'rew_hist' : rew_hist, 
            'disc_rewards' : discounted_rewards,
            'policy' : policy_output, 'value' : value_output, 
            'model_regrets' : model_regrets}    
 

    

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    

    
class MyDDP(DDP):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
    

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        args: dict,
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.optimizer = optimizer
        self.args = args
        self.model = MyDDP(self.model, device_ids=[gpu_id], find_unused_parameters=True)
        
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr = args['LR'] / args['N_mini_batches'] )
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr = args['LR'] / args['N_mini_batches'], momentum = 0.9)
        
        self.val_coeff_list = [0.1] * args['train_steps']
        self.entropy_coeff_list = dynamic_coeff(start = 0.1, finish = 0.0, total_len = args['train_steps'], progress_pct = 0.6)
        self.lambda_list = dynamic_coeff(start = 0.7, finish = 1.0, total_len = args['train_steps'], progress_pct = 0.6)
        
    
    
    def train_step(self, cur_args, loss_computer):    
        def get_actions_log_probs(act_hist, policy_output):
            log_probs = nn.LogSoftmax(dim = 2)(policy_output)            
            idx = (act_hist[:, 1:]).long().view(act_hist.shape[0], -1, 1).cuda().detach()
            selected_log_probs = torch.gather(log_probs, 2, idx).squeeze(2)
            return selected_log_probs

        major_device = get_major_device(self.model)
        rollouts = get_rollouts(self.model, cur_args)
        entropy_loss = loss_computer.entropy_loss(rollouts['policy'])
        value_loss = loss_computer.value_loss(rollouts['value'], 
                                              disc_rewards = rollouts['disc_rewards'][:, 1:]) #.to(major_device))
        chosen_actions_logs = get_actions_log_probs(rollouts['act_hist'].detach(), rollouts['policy'])
        policy_loss = loss_computer.policy_loss_GAE(chosen_actions_logs, 
                                                    rewards = rollouts['rew_hist'][:, 1:], #.to(major_device), 
                                                    values = rollouts['value'].detach().squeeze(2), 
                                                   lambdaa = cur_args['lambdaa_coeff'])    
        (policy_loss + cur_args['value_coeff'] * value_loss + cur_args['entropy_coeff'] * entropy_loss).backward()
        return value_loss, policy_loss, entropy_loss, rollouts['model_regrets'].detach()
            

    
    def train(self):
        loss_computer = BanditLossComputer(self.args)
        policy_regrets = []
    
        for t in tqdm(range(self.args['train_steps'])):
            
            total_value_loss, total_policy_loss, total_regret, total_entropy_loss = 0, 0, 0, 0

            cur_args = self.args.copy()
            cur_args['entropy_coeff'] = self.entropy_coeff_list[t]
            cur_args['value_coeff'] = self.val_coeff_list[t]
            cur_args['lambdaa_coeff'] = self.lambda_list[t]
            
            self.optimizer.zero_grad()
            value_loss, policy_loss, entropy_loss, model_regrets = self.train_step(cur_args, loss_computer)
            self.optimizer.step()
            
            total_regret += torch.mean(torch.sum(model_regrets, axis = 1)).item()
            total_entropy_loss += entropy_loss.detach().item()
            total_value_loss += value_loss.detach().item()
            total_policy_loss += policy_loss.detach().item()
            policy_regrets.append(total_regret / self.args['N_pergpu'])
            
            print(f"step = {t}, gpu_id = {self.gpu_id}, policy regret = {torch.mean(torch.sum(model_regrets, axis = 1)).item()}")
        
            if self.gpu_id == 0 and (t + 1) % self.args['save_every_iter'] == 0:
                print(f'step = {t}, saving model in gpu = 0')
    

    
def load_train_objs(args):
    model = TransformerModelSymmetric(n_arms = args['n_arms'], n_dims = 2, 
                                        n_embd = args['n_embed'], n_layer = args['n_layer'], 
                                        n_head = args['n_head'], symmetric = args['symmetric'], 
                                        use_cache = args['use_cache'])
    optimizer = torch.optim.Adam(model.parameters(), lr = args['LR'] / args['N_mini_batches'] )
    
    return model, optimizer



def main(rank: int, world_size: int, args : dict):
    ddp_setup(rank, world_size)
    model, optimizer = load_train_objs(args)
    
    trainer = Trainer(model, optimizer, rank, args)
    trainer.train()
    destroy_process_group()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process YAML file with command-line arguments")
    parser.add_argument("yaml_file", help="Path to the YAML file")
    args = sys.argv[1:]
    args = parser.parse_args(args)
    
    with open(args.yaml_file, 'r') as yaml_file:
        args = yaml.safe_load(yaml_file)
        
    print(f"Run config dict: {args}")
    args = args['run_config']
    
    # world_size = torch.cuda.device_count()
    world_size = args['n_gpus']
    mp.spawn(main, args=(world_size, args), nprocs=world_size)