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





from env_MAB import *
from my_algorithms import * 
from util import *
from transformer import *
from train_utils import *


torch.autograd.set_detect_anomaly(True)
working_dir = "/n/home03/vladpetrov/thesis"



    
    
def get_rollouts(model, args):
    '''
    this is where interaction with the environment (bandits) is hapenning:
    1) sample B (batch size) bandits from Uniform prior
    2) iteratively interact with each of them from t = 1 to T (horizon) and observe rewards r_t
    each time transformer sees all the past trajectory up to moment t, so it's able to learn from this context what actions to make
    3) final trajectories, rewards, policy and value outputs are returned in the end at t = T
    '''
    mini_b_size = args['mini_b_size']
    n_arms, dim_size = model.n_arms, 1
    
    # used to have implementation of bandits as a Python class
    # makes code look nicer, but also slows down the training process because policy and bandits have to interact with each other at every time step t, 
    # i.e. we have to constantly communicate between CPU and GPU

    # bandits = [MAB(T = args['horizon'], mu_list = mus[i]) for i in range(mini_b_size)]
    # first_actions = torch.tensor(np.random.choice(n_arms, mini_b_size))
    # first_rewards = torch.tensor([bandits[i].pull(first_actions[i]) for i in range(mini_b_size)])
    

    # here: changed to bandits being fully stored on cuda
    # quite improves learning speed
    mus = sample_mus(mini_b_size, n_arms, regime = args['mus_regime'])
    mus = torch.tensor(mus).cuda()    
    
    def pull_arms(mus, chosen_actions):         
        random_draws = torch.rand((chosen_actions.shape[0], 1), dtype=float, device='cuda')
        chosen_mus = torch.gather(mus, dim = 1, index = chosen_actions)
        rewards = (random_draws < chosen_mus).float()
        return rewards - 0.5
    

    # first actions are randomly generated
    first_actions = torch.tensor(np.random.choice(n_arms, mini_b_size)).view(mini_b_size, -1).cuda()
    first_rewards = pull_arms(mus, first_actions).view(mini_b_size, -1).cuda()
    
    act_hist = first_actions.view(mini_b_size, -1).long()
    rew_hist = first_rewards.view(mini_b_size, -1).float()
    model_regrets = torch.tensor([]).view(mini_b_size, 0).float().cuda()
    discounted_rewards = first_rewards.view(mini_b_size, -1).float().cuda()
    chosen_mus = torch.gather(mus, 1, first_actions)
    
    del first_actions, first_rewards
    
    
    discount = 1
    past = None
    if args['print_memory_usage']:
        output_memory_usage(0)
        
    
    # t = 1 .... horizon
    for seq_len in range(args['horizon']):
        discount *= args['gamma']
        
        output_dict = model(
                act_hist.view(mini_b_size, seq_len + 1, dim_size),
                rew_hist.view(mini_b_size, seq_len + 1, dim_size),
            )
        
#         # use past attention keys for optimization
#         if (past is not None) and (seq_len < args['horizon'] - 1):
#             output_dict = model(
#                 act_hist.view(mini_b_size, seq_len + 1, dim_size)[:, -1:, :],
#                 rew_hist.view(mini_b_size, seq_len + 1, dim_size)[:, -1:, :],
#                 past = past,
#             )
#         else:
#             output_dict = model(
#                 act_hist.view(mini_b_size, seq_len + 1, dim_size),
#                 rew_hist.view(mini_b_size, seq_len + 1, dim_size),
#             )
        
        if seq_len < args['horizon'] - 1:
            policy_output = output_dict['policy'].detach()
        else:
            policy_output, value_output = output_dict['policy'], output_dict['value']

        action_distribution = Categorical(logits = policy_output[:, -1, :].detach())
        new_actions = action_distribution.sample().detach().view(act_hist.shape[0], -1).long()
        new_rewards = pull_arms(mus, new_actions).detach()

        act_hist = torch.cat((act_hist, new_actions), 1).detach()
        rew_hist = torch.cat((rew_hist, new_rewards), 1).detach()

        del new_actions, new_rewards, action_distribution, output_dict
        if seq_len < args['horizon'] - 1:
            del policy_output
            
        if args['print_memory_usage']:
            output_memory_usage(seq_len + 1)
            
    
    discounted_rewards = rew_hist.detach() * torch.vander(torch.tensor(
        [args['gamma']] * rew_hist.shape[0]),  N = rew_hist.shape[1], increasing = True).cuda()
    chosen_mus = torch.gather(mus, 1, act_hist).detach()
    model_regrets = torch.max(mus, dim = 1, keepdim = True)[0] - chosen_mus
    model_regrets = model_regrets * torch.vander(torch.tensor(
        [args['gamma']] * model_regrets.shape[0]), N = model_regrets.shape[1], increasing = True).cuda()
    model_regrets = model_regrets[:, 1:].detach()

    
    symmetry_variance = None
    if 'lam_symmetry' in args.keys() and args['lam_symmetry'] >= 0:
        FRACTION = 1 / model.n_arms
        num_samples = int(FRACTION * act_hist.shape[0])
        sampled_indices = torch.randperm(act_hist.shape[0])[:num_samples]
        symmetry_variance = model(
                act_hist[sampled_indices].view(num_samples, args['horizon'] + 1, dim_size).detach(),
                rew_hist[sampled_indices].view(num_samples, args['horizon'] + 1, dim_size).detach(),
                symmetric_pass = True
            )['symmetry_variance']
        
#     output_memory_usage(args['horizon'])
                
    return {'act_hist': act_hist, 'rew_hist' : rew_hist, 
            'disc_rewards' : discounted_rewards,
            'policy' : policy_output, 'value' : value_output, 
            'model_regrets' : model_regrets, 
            'chosen_mus' : chosen_mus,
            'symmetry_variance' : symmetry_variance}
 
    
def train_step(model, args, loss_computer, print_memory = False):
    '''
    this is where major learning is happening:
    1) get trajectory rollouts as specified in get_rollouts() function
    2) compute and return different loss components using loss_computer class. 
    These include regular REINFORCE policy loss, value loss, entropy regularization; 
    I incapsulated all loss computations in the class BanditLossComputer for convenience, which can be found in train_utils file

    symmetry regularization (variance) is implemented separately inside the get_rollouts() function
    '''
    
    def get_actions_log_probs(act_hist, policy_output):
        log_probs = nn.LogSoftmax(dim = 2)(policy_output)            
        idx = (act_hist[:, 1:]).long().view(act_hist.shape[0], -1, 1).cuda().detach()
        selected_log_probs = torch.gather(log_probs, 2, idx).squeeze(2)
        return selected_log_probs
    
    
    
    rollouts = get_rollouts(model, args)    
    entropy_loss = loss_computer.entropy_loss(rollouts['policy'])
    value_loss = loss_computer.value_loss(rollouts['value'], 
                                          disc_rewards = rollouts['disc_rewards'][:, 1:]) #.to(major_device))
    
    chosen_actions_logs = get_actions_log_probs(rollouts['act_hist'].detach(), rollouts['policy'])
    policy_loss = loss_computer.policy_loss_GAE(chosen_actions_logs, 
                                                rewards = rollouts['rew_hist'][:, 1:], #.to(major_device), 
                                                values = rollouts['value'].detach().squeeze(2), 
                                               lambdaa = args['lambdaa_coeff'])
    
    if 'lam_symmetry' in args.keys() and args['lam_symmetry'] >= 0:

        ave_variance = torch.mean(rollouts['symmetry_variance'], dim = -1)
        ave_variance = ave_variance * torch.vander(torch.tensor(
            [args['gamma']] * ave_variance.shape[0]), N = ave_variance.shape[1], increasing = True).cuda()
        symmetry_loss = torch.mean( torch.sum(ave_variance, dim = -1) )
            
        (policy_loss + args['value_coeff'] * value_loss +\
         args['entropy_coeff'] * entropy_loss + args['lam_symmetry'] * symmetry_loss).backward()
    else:
        symmetry_loss = torch.tensor(-1).cuda()
        (policy_loss + args['value_coeff'] * value_loss + args['entropy_coeff'] * entropy_loss).backward()
    
    
    result = {
        'value_loss' : value_loss, 'policy_loss' : policy_loss, 'entropy_loss' : entropy_loss, 
        'model_regrets' : rollouts['model_regrets'].detach(), 
        'chosen_mus' : rollouts['chosen_mus'].detach(),
        'symmetry_loss' : symmetry_loss,
    }
    return result


    

def train(model, args):    
    
    optimizer = torch.optim.Adam(model.parameters(), lr = args['LR'] / args['N_mini_batches'] )
    scheduler = create_simple_scheduler(optimizer, args['train_steps'], args['warmup_steps'])

    loss_computer = BanditLossComputer(args)
    policy_regrets = []

    # load model state in case we continue training from some point
    if 'model_state_path' in args.keys():
        
        print(f'continue training model')
        
        state = torch.load(args['model_state_path'])
        
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        scheduler.load_state_dict(state['scheduler_state_dict'])
        
        start_step = state['train_step'] + 1
        val_coeff_list, entropy_coeff_list, lambda_list = state['val_coeff_list'], state['entropy_coeff_list'], state['lambda_list']
        
        policy_regrets = state['policy_regrets']
        print(f"Downloaded model from {args['model_state_path']}\n")
        
        if len(lambda_list) < args['train_steps']:
            print("post-training; adding final (const) values to coefficients; LR will be constant")
            remaining = args['train_steps'] - len(lambda_list)
            val_coeff_list += [0.1] * remaining
            entropy_coeff_list += [0.0] * remaining
            lambda_list += [1.0] * remaining
            scheduler = create_const_scheduler(optimizer, multiplier = 1)
            
            if 'lam_symmetry' in args.keys():
                lam_symmetry_list = [ args['lam_symmetry'] ] * args['train_steps']
            
            
    else:
        print(f'training new model: {model.name}\n')
        
        start_step = 0

        # define different hyperparameter schedules, e.g., annealing entropy regularization coefficients
        val_coeff_list = [0.1] * args['train_steps']
        start_entropy_coeff = 0.5 if not('start_entropy_coeff' in args.keys()) else args['start_entropy_coeff']
        finish_entropy_coeff = 0.0 if not('finish_entropy_coeff' in args.keys()) else args['finish_entropy_coeff']
        
        entropy_coeff_list = dynamic_coeff(start = start_entropy_coeff, finish = finish_entropy_coeff, 
                                           total_len = args['train_steps'], progress_pct = 0.5)
        lambda_list = dynamic_coeff(start = 0.3, finish = 1.0, total_len = args['train_steps'], progress_pct = 0.5)
        
        if 'lam_symmetry' in args.keys():
            lam_symmetry_list = [ args['lam_symmetry'] ] * args['train_steps']
                
    
    for t in tqdm(range(start_step, args['train_steps']), mininterval = 0, miniters = 1):
        
        optimizer.zero_grad()

        total_value_loss, total_policy_loss, total_regret, total_entropy_loss = 0, 0, 0, 0
        total_reward = 0
        total_symmetry_loss = 0
        
        cur_args = args.copy()
        cur_args['entropy_coeff'] = entropy_coeff_list[t]
        cur_args['value_coeff'] = val_coeff_list[t]
        cur_args['lambdaa_coeff'] = lambda_list[t]
        
        if 'lam_symmetry' in args.keys():
            cur_args['lam_symmetry'] = lam_symmetry_list[t]
        
        # perform N_mini_batches times graident accumulation 
        for _ in range(args['N_mini_batches']):

            result = train_step(model, cur_args, loss_computer)
            total_entropy_loss += result['entropy_loss'].detach().item()
            total_value_loss += result['value_loss'].detach().item()
            total_policy_loss += result['policy_loss'].detach().item()
            total_regret += torch.mean(torch.sum(result['model_regrets'], axis = 1)).item() # true loss that we care about
            total_reward += torch.mean(torch.sum(result['chosen_mus'], axis = 1)).item()
            total_symmetry_loss += result['symmetry_loss'].detach().item()
            
        
        optimizer.step()
        scheduler.step()
        
        policy_regrets.append(total_regret / args['N_mini_batches'])
        print(f"step = {t}, c_entrop = {np.round(entropy_coeff_list[t], 3)}, c_val = {np.round(val_coeff_list[t], 3)}, lambda = {np.round(lambda_list[t], 3)}")
        print(f"lr rate = {scheduler.get_last_lr()[0]}, policy regret = {policy_regrets[-1]}")
        
        if wandb.run is not None:
            
            wandb.log({
                "policy_loss": total_policy_loss / args['N_mini_batches'],
                "entropy_loss": total_entropy_loss / args['N_mini_batches'],
                "value_loss": total_value_loss / args['N_mini_batches'],
                "policy_regret" : total_regret / args['N_mini_batches'],
                "reward" : total_reward / args['N_mini_batches'],
                "symmetry_loss": total_symmetry_loss / args['N_mini_batches'],
                "lr_rate" : float(scheduler.get_last_lr()[0]),
                },
                step = t
            )
        


        if 'save_every_iter' in args.keys() and (t + 1) % args['save_every_iter'] == 0:
            training_state = {
                'model_state_dict' : model.state_dict(), 'train_step' : t, 
                'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),
                'val_coeff_list' : val_coeff_list, 'entropy_coeff_list' : entropy_coeff_list, 'lambda_list': lambda_list,
                'policy_regrets' : policy_regrets,
            }
            torch.save(training_state, args['model_save_path'])
            print(f'saved model to {args["model_save_path"]}')
        
        
        if 'run_experiment_every' in args.keys() and (t + 1) % args['run_experiment_every'] == 0:
            
            run_experiment(model, horizon = args['horizon'], gamma = args['gamma'], b_size = 100)
            
            
            
            
class MyDataParallel(nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)



def main(args):
    parser = argparse.ArgumentParser(description="Process YAML file with command-line arguments")
    parser.add_argument("yaml_file", help="Path to the YAML file")
    args = parser.parse_args(args)
    
    with open(args.yaml_file, 'r') as yaml_file:
        args = yaml.safe_load(yaml_file)
        
    print(f"Initial config dictionary: {args}")
    
    if 'wandb_config' in args.keys():
        wandb_config = args['wandb_config']
        if 'name' not in wandb_config.keys():
            wandb_config['name'] = None
        if 'resume_run' not in wandb_config.keys():
            wandb_config['resume_run'] = False
            
        args = args['run_config']

        run = wandb.init(
            project = wandb_config['project'],
            name = wandb_config['name'],
            config = args,
            resume = wandb_config['resume_run'],
        )
        RUN_NAME = run.name
        print(f'initialized run {RUN_NAME}')
    else:
        print(f'NB: no wandb')
        RUN_NAME = "test_run"
        args = args['run_config']
        


    
    if 'symmetric' in args.keys() and args['symmetric'] is True:
        assert not ('lam_symmetry' in args.keys())
        
        model = TransformerModelSymmetric(n_arms = args['n_arms'], n_dims = 2, 
                                            n_embd = args['n_embd'], n_layer = args['n_layer'], 
                                            n_head = args['n_head'], use_cache = args['use_cache'], 
                                            symmetric = True, max_input_len = 128).to('cuda')
    else:
        
        model = TransformerModelReg(args, max_input_len = 128).to('cuda')


    model = MyDataParallel(model, device_ids = np.arange(args['n_gpus']))

    
    if not ('model_state_path' in args.keys()):
        print(f"training new model:")
        args['model_save_path'] = f"{working_dir}/models/run_{RUN_NAME}"
    else:
        print(f"continuing running model: {args['model_state_path']}")
        args['model_save_path'] = f"{working_dir}/models/run_{RUN_NAME}"
    
    train(model, args)
    wandb.finish()
    

    
if __name__ == "__main__":
    main(sys.argv[1:])
    