import numpy as np
import random
import sys
import csv
import pandas as pd
import matplotlib.pyplot as plt


from tqdm import tqdm

from torch.distributions.categorical import Categorical
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from env_MAB import *
from my_algorithms import *
from util import * 
from transformer import *




def run_experiment(model, horizon, gamma, mini_b_size, N_mini_batches, mu_list,
                        include_gittins = False, print_memory = False, dim_size = 1):
    '''outputs model's action for a given list of mus 
    optionally, compares with gittins index'''
    

    fig, axs = plt.subplots(len(mu_list))
    fig.set_figwidth(10)
    fig.set_figheight(6 * len(mu_list))
    
    n_arms = model.n_arms
    

    for index, mu_pair in enumerate(mu_list):
        
        model_actions = torch.tensor([]).view(horizon + 1, 0)
        gittins_actions = torch.tensor([]).view(horizon + 1, 0)
        model_regrets, gittins_regrets = [], []
        
        for _ in range(N_mini_batches):
            
            mus = np.array([mu_pair] * mini_b_size).astype(float)
            bandits = [MAB(T = horizon, mu_list = mus[i]) for i in range(mini_b_size)]
            gittins = [Gittins_index(bandits[i], gamma = gamma, N = horizon) for i in range(mini_b_size)]

            first_actions = torch.tensor(np.random.choice(n_arms, mini_b_size)).cuda()
            first_rewards = torch.tensor([bandits[i].pull(first_actions[i]) for i in range(mini_b_size)]).cuda()
            act_hist = first_actions.view(mini_b_size, -1).float()
            rew_hist = first_rewards.view(mini_b_size, -1).float()
            
            if include_gittins:
                git_act_hist = torch.clone(act_hist)
                for i in range(mini_b_size):
                    # updates gittins statistic for the arm pulled at step 0
                    gittins[i].compute_gittins_index(first_actions[i])
                    
            
            cur_model_regrets = torch.tensor([]).view(mini_b_size, 0).float()
            cur_gittins_regrets = torch.tensor([]).view(mini_b_size, 0).float()
               
            discount = 1
            for seq_len in range(bandits[0].get_T()):
                discount *= gamma
                
                if include_gittins:
                    gittens_actions = [gittins[i].choose_gittins_arm() for i in range(mini_b_size)]
                    git_act_hist = torch.cat((git_act_hist, torch.tensor(gittens_actions).view(mini_b_size, 1).cuda()), 1)
                    new_gittins_regrets = discount * np.array([np.max(mus[i]) - mus[i][gittens_actions[i]] 
                                               for i in range(mini_b_size)]).reshape(mini_b_size, 1)
                    cur_gittins_regrets = torch.cat((cur_gittins_regrets, torch.tensor(new_gittins_regrets)), 1)
                    
                
                output = model(
                  act_hist.view(mini_b_size, seq_len + 1, dim_size),
                  rew_hist.view(mini_b_size, seq_len + 1, dim_size))

                action_distribution = Categorical(logits = output[:, -1, :].detach())
                new_actions = action_distribution.sample().cpu().detach().numpy().reshape(-1, 1)
                new_rewards = np.array([bandits[i].pull(new_actions[i]) for i in range(len(new_actions))]).reshape(-1, 1)
                
                if include_gittins:
                    for i in range(len(new_actions)):
                        gittins[i].compute_gittins_index(new_actions[i][0])

                act_hist = torch.cat((act_hist, torch.tensor(new_actions).cuda()), 1)
                rew_hist = torch.cat((rew_hist, torch.tensor(new_rewards).cuda()), 1)
                
                new_model_regrets = discount * np.array([np.max(mus[i]) - mus[i][new_actions[i]] 
                                                 for i in range(mini_b_size)]).reshape(mini_b_size, 1)
                cur_model_regrets = torch.cat((cur_model_regrets, torch.tensor(new_model_regrets)), 1)
                

                del new_actions, new_rewards, action_distribution, new_model_regrets
                if print_memory:
                    print(f'memory usage for seq_len = {seq_len}')
                    output_cuda_memory()
                torch.cuda.empty_cache()


            model_actions = torch.cat((model_actions, torch.mean(act_hist, axis = 0).view(horizon + 1, 1).cpu()), 1)
            model_regrets.append(torch.mean(torch.sum(cur_model_regrets, axis = 1)).item())
            
            if include_gittins:
                gittins_actions = torch.cat((gittins_actions, torch.mean(git_act_hist, axis = 0).view(horizon + 1, 1).cpu()), 1)
                gittins_regrets.append(torch.mean(torch.sum(cur_gittins_regrets, axis = 1)).item())
            

        df = pd.DataFrame({'model_actions' : torch.mean(model_actions, axis = 1).numpy()})
        print(f"mu_pair = {mu_pair}:")
        print(f"avg model regret = {np.mean(model_regrets)}, std = {np.std(model_regrets)}")
        
        if include_gittins:
            df['gittins_actions'] = torch.mean(gittins_actions, axis = 1).numpy()
            print(f"avg gittins regret = {np.mean(gittins_regrets)}, std = {np.std(gittins_regrets)}")
            
            
        df.plot(ax = axs[index])
        axs[index].set_xlabel(f'mu_pair = {mu_pair}')
            
            
    
    plt.show()
   
    
    
  
    

    
def run_experiment_gittins(horizon = 20, b_size = 128, gamma = 1, dim_size = 1, mus_regime = 'uniform'):
    '''
    intends to estimate gittins index regret for the specific task (horizon, discount gamma, etc.)
    '''
    
    mus = sample_mus(b_size, n_arms = 2, regime = mus_regime)

    bandits = [MAB(T = horizon, mu_list = mus[i]) for i in range(b_size)]
    
    gittins = [Gittins_index(bandits[i], gamma = gamma, N = horizon) for i in range(b_size)]
    gittins_regrets = torch.tensor([]).view(b_size, 0).float()
    
    first_actions = [gittins[i].play_one_step() for i in range(b_size)]

    discount = 1
    for seq_len in range(bandits[0].get_T()):
        discount *= gamma
        
        gittens_actions = [gittins[i].play_one_step() for i in range(b_size)]
        new_gittins_regrets = discount * np.array([np.max(mus[i]) - mus[i][gittens_actions[i]] 
                                           for i in range(b_size)]).reshape(b_size, 1)
        gittins_regrets = torch.cat((gittins_regrets, torch.tensor(new_gittins_regrets)), 1)

    return torch.mean(torch.sum(gittins_regrets, axis = 1)).item()
    

    
    
    
    

# older version 
def run_experiment_upd(model, horizon = 25, gamma = 1, b_size = 128, dim_size = 1, 
                       my_strategies = False, print_mode = False, model_decision = 'sample',
                mu_list = [
                [0, 1], [0.1, 0.9], [0.2, 0.8], [0.4, 0.6],
                [0.6, 0.4], [0.8, 0.2], [0.9, 0.1], [1, 0],
                [0.1, 0.2], [0.2, 0.1], [0.7, 0.8], [0.8, 0.7]]):
    
    
    fig, axs = plt.subplots(len(mu_list))
    fig.set_figwidth(10)
    fig.set_figheight(6 * len(mu_list))
    
    n_arms = model.n_arms

    for index, mu_pair in enumerate(mu_list):

        mus = np.array([mu_pair] * b_size).astype(float)
        bandits = [MAB(T = horizon, mu_list = mus[i]) for i in range(b_size)]
        gittins = [Gittins_index(bandits[i], gamma = gamma, N = horizon) for i in range(b_size)]
        
        model_regrets = torch.tensor([]).view(b_size, 0).float()
        gittins_regrets = torch.tensor([]).view(b_size, 0).float()        
        
        first_actions = torch.tensor(np.random.choice(n_arms, b_size)).cuda()
        first_rewards = torch.tensor([bandits[i].pull(first_actions[i]) for i in range(b_size)]).cuda()
        
        if my_strategies:
            gittens_choices = [torch.mean(first_actions.float()).item()]
            for i in range(b_size):
                gittins[i].compute_gittins_index(first_actions[i])
            
        
        act_hist = first_actions.view(b_size, -1).float()
        rew_hist = first_rewards.view(b_size, -1).float()

        discount = 1
        for seq_len in range(bandits[0].get_T()):
            discount *= gamma
            
            if my_strategies:
                gittens_actions = [gittins[i].choose_gittins_arm() for i in range(b_size)]
                
                new_gittins_regrets = discount * np.array([np.max(mus[i]) - mus[i][gittens_actions[i]] 
                                           for i in range(b_size)]).reshape(b_size, 1)
                gittins_regrets = torch.cat((gittins_regrets, torch.tensor(new_gittins_regrets)), 1)
                gittens_choices.append(np.mean(gittens_actions))
                
                
            output = model(
              act_hist.view(b_size, seq_len + 1, dim_size),
              rew_hist.view(b_size, seq_len + 1, dim_size),
            )
            
            if model_decision == 'sample':
                action_distribution = Categorical(logits = output[:, -1, :].detach())
                new_actions = action_distribution.sample().cpu().detach().numpy().reshape(-1, 1)
            elif model_decision == 'argmax':
                new_actions = torch.argmax(output[:, -1, :].detach(), dim = 1).cpu().detach().numpy().reshape(-1, 1)
                
            new_rewards = np.array([bandits[i].pull(new_actions[i]) for i in range(len(new_actions))]).reshape(-1, 1)
            
            if my_strategies:
                for i in range(len(new_actions)):
                    gittins[i].compute_gittins_index(new_actions[i][0])

            act_hist = torch.cat((act_hist, torch.tensor(new_actions).cuda()), 1)
            rew_hist = torch.cat((rew_hist, torch.tensor(new_rewards).cuda()), 1)
            
            new_model_regrets = discount * np.array([np.max(mus[i]) - mus[i][new_actions[i]] 
                                                 for i in range(b_size)]).reshape(b_size, 1)
            model_regrets = torch.cat((model_regrets, torch.tensor(new_model_regrets)), 1)
            
            del new_actions, new_rewards, new_model_regrets
            
            if print_mode:
                print(f'memory usage for seq_len = {seq_len}')
                output_cuda_memory()
            
            torch.cuda.empty_cache()

    
        chosen_actions = torch.mean(act_hist, axis = 0)

        print(f"mu_pair = {mu_pair}:")
        print(f"avg model regret = {torch.mean(torch.sum(model_regrets, axis = 1)).item()}")
        if my_strategies:
            print(f"avg gittins regret = {torch.mean(torch.sum(gittins_regrets, axis = 1)).item()}")
        print(f"------")

        df = pd.DataFrame({'model_actions' : chosen_actions.cpu().detach().numpy()})
        if my_strategies:
            df['gittins'] = gittens_choices
            
        df.plot(ax = axs[index])
        axs[index].set_xlabel(f'mu_pair = {mu_pair}')
    
    plt.show()
    
    
    
    
    
# def run_experiment_long(model, horizon, mini_b_size, N_mini_batches, mu_list, dim_size = 1, 
#                         my_strategies = False, print_memory = False):
    
#     '''
#     longer experiment to get better estimates of model's performance (regret & actions)
#     '''

#     fig, axs = plt.subplots(len(mu_list))
#     fig.set_figwidth(10)
#     fig.set_figheight(6 * len(mu_list))
    
#     n_arms = model.n_arms
    

#     for index, mu_pair in enumerate(mu_list):
        
#         model_regrets = []
#         model_actions = torch.tensor([]).view(horizon + 1, 0)
        
#         for _ in range(N_mini_batches):

#             mus = np.array([mu_pair] * mini_b_size).astype(float)
#             bandits = [MAB(T = horizon, mu_list = mus[i]) for i in range(mini_b_size)]

#             first_actions = torch.tensor(np.random.choice(n_arms, mini_b_size)).cuda()
#             first_rewards = torch.tensor([bandits[i].pull(first_actions[i]) for i in range(mini_b_size)]).cuda()
#             act_hist = first_actions.view(mini_b_size, -1).float()
#             rew_hist = first_rewards.view(mini_b_size, -1).float()

#             for seq_len in range(bandits[0].get_T()):
#                 output = model(
#                   act_hist.view(mini_b_size, seq_len + 1, dim_size),
#                   rew_hist.view(mini_b_size, seq_len + 1, dim_size),)

#                 action_distribution = Categorical(logits = output[:, -1, :].detach())
#                 new_actions = action_distribution.sample().cpu().detach().numpy().reshape(-1, 1) # index represents the arm we pull
#                 new_rewards = np.array([bandits[i].pull(new_actions[i]) for i in range(len(new_actions))]).reshape(-1, 1)

#                 act_hist = torch.cat((act_hist, torch.tensor(new_actions).cuda()), 1)
#                 rew_hist = torch.cat((rew_hist, torch.tensor(new_rewards).cuda()), 1)

#                 del new_actions, new_rewards, action_distribution
#                 if print_memory:
#                     print(f'memory usage for seq_len = {seq_len}')
#                     output_cuda_memory()
#                 torch.cuda.empty_cache()


#             chosen_actions = torch.mean(act_hist, axis = 0).view(horizon + 1, 1).cpu()
#             model_actions = torch.cat((model_actions, chosen_actions), 1)
#             model_regrets.append( np.array([np.mean(bandits[i].get_regrets()) for i in range(mini_b_size)]).mean() )            
            

#         df = pd.DataFrame({'model_actions' : torch.mean(model_actions, axis = 1).numpy()})
#         df.plot(ax = axs[index])
#         axs[index].set_xlabel(f'mu_pair = {mu_pair}')
        
#         print(f"mu_pair = {mu_pair}:")
#         print(f"avg model regret = {np.mean(model_regrets)}")
            
    
#     plt.show()
    
    
    


# def run_experiment_gittins(model, horizon = 20, b_size = 128, gamma = 1, dim_size = 1, mus_regime = 'uniform'):

#     mus = sample_mus(b_size, n_arms, regime = mus_regime)
#     n_arms = model.n_arms

#     bandits = [MAB(T = horizon, mu_list = mus[i]) for i in range(b_size)]
    
#     gittins = [Gittins_index(bandits[i], gamma = gamma) for i in range(b_size)]
#     gittins_regrets = torch.tensor([]).view(b_size, 0).float()


#     first_actions = torch.tensor(np.random.choice(n_arms, b_size))
#     first_rewards = torch.tensor([bandits[i].pull(first_actions[i]) for i in range(b_size)])
#     act_hist, rew_hist = first_actions.view(b_size, -1).float(), first_rewards.view(b_size, -1).float()
#     model_regrets = torch.tensor([]).view(b_size, 0).float()


#     discount = 1
#     for seq_len in range(bandits[0].get_T()):
#         discount *= gamma
        
#         gittens_actions = [gittins[i].choose_gittins_arm() for i in range(b_size)]
#         new_gittins_regrets = discount * np.array([np.max(mus[i]) - mus[i][gittens_actions[i]] 
#                                            for i in range(b_size)]).reshape(b_size, 1)
#         gittins_regrets = torch.cat((gittins_regrets, torch.tensor(new_gittins_regrets)), 1)

        
#         output = model(
#           act_hist.view(b_size, seq_len + 1, dim_size).cuda(),
#           rew_hist.view(b_size, seq_len + 1, dim_size).cuda(),
#         )

#         action_distribution = Categorical(logits = output[:, -1, :].detach())
#         new_actions = action_distribution.sample().cpu().detach().numpy().reshape(-1, 1) # index represents the arm we pull
#         new_rewards = np.array([bandits[i].pull(new_actions[i]) for i in range(len(new_actions))]).reshape(-1, 1)
#         act_hist = torch.cat((act_hist, torch.tensor(new_actions)), 1).detach()
#         rew_hist = torch.cat((rew_hist, torch.tensor(new_rewards)), 1).detach()
        
#         new_model_regrets = discount * np.array([np.max(mus[i]) - mus[i][new_actions[i]] 
#                                                  for i in range(b_size)]).reshape(b_size, 1)
#         model_regrets = torch.cat((model_regrets, torch.tensor(new_model_regrets)), 1)
        
#         del new_actions, new_rewards, action_distribution


#     return {'model' : torch.mean(model_regrets).item(),
#             'gittens' : torch.mean(gittins_regrets).item()}



# def run_experiment_upd(model, new_T = 25, b_size = 128, dim_size = 1, my_strategies = False, print_mode = False,
#                 mu_list = [
#                 [0, 1], [0.1, 0.9], [0.2, 0.8], [0.4, 0.6],
#                 [0.6, 0.4], [0.8, 0.2], [0.9, 0.1], [1, 0],
#                 [0.1, 0.2], [0.2, 0.1], [0.7, 0.8], [0.8, 0.7]]):
    
    
#     fig, axs = plt.subplots(len(mu_list))
#     fig.set_figwidth(10)
#     fig.set_figheight(6 * len(mu_list))
    
#     n_arms = model.n_arms

#     for index, mu_pair in enumerate(mu_list):

#         mus = np.array([mu_pair] * b_size).astype(float)
#         bandits = [MAB(T = new_T, mu_list = mus[i]) for i in range(b_size)]
#         gittins = [Gittins_index(bandits[i]) for i in range(b_size)]
        
#         model_regrets = torch.tensor([]).view(b_size, 0).float()
#         gittins_regrets = torch.tensor([]).view(b_size, 0).float()
        
#         #thompson = [Thompson_sampling(bandits[i]) for i in range(b_size)]
#         #thompson_choices = [np.mean( [thompson[i].choose_thompson_arm() for i in range(b_size)] )]
#         #gittens_choices = [np.mean( [gittins[i].choose_gittins_arm() for i in range(b_size)] )]
#         #thompson_rewards, gittens_rewards = [], []
        
        
#         # first_actions = torch.tensor(np.random.choice(n_arms, b_size)).cuda()
#         first_actions = torch.tensor([gittins[i].play_one_step() for i in range(b_size)]).cuda() #uniform prior
#         gittens_choices = [torch.mean(first_actions).item()]
#         first_rewards = torch.tensor([bandits[i].pull(first_actions[i]) for i in range(b_size)]).cuda()
        
        
#         act_hist = first_actions.view(b_size, -1).float()
#         rew_hist = first_rewards.view(b_size, -1).float()

#         discount = 1
#         for seq_len in range(bandits[0].get_T()):
#             discount *= gamma
            
#             if my_strategies:
                
#                 gittens_actions = [gittins[i].choose_gittins_arm() for i in range(b_size)]
                
#                 new_gittins_regrets = discount * np.array([np.max(mus[i]) - mus[i][gittens_actions[i]] 
#                                            for i in range(b_size)]).reshape(b_size, 1)
#                 gittins_regrets = torch.cat((gittins_regrets, torch.tensor(new_gittins_regrets)), 1)
#                 gittens_choices.append(np.mean(gittens_actions))
                
#                 #thompson_choices.append(np.mean( [ thompson[i].choose_thompson_arm() for i in range(b_size) ] ))
#                 #thompson_rewards.append((1 - thompson_choices[-1]) * mu_pair[0] + thompson_choices[-1] * mu_pair[1])
#                 #gittens_rewards.append((1 - gittens_choices[-1]) * mu_pair[0] + gittens_choices[-1] * mu_pair[1])
                
                
#             output = model(
#               act_hist.view(b_size, seq_len + 1, dim_size),
#               rew_hist.view(b_size, seq_len + 1, dim_size),
#             )

#             action_distribution = Categorical(logits = output[:, -1, :].detach())
#             new_actions = action_distribution.sample().cpu().detach().numpy().reshape(-1, 1) # index represents the arm we pull
#             new_rewards = np.array([bandits[i].pull(new_actions[i]) for i in range(len(new_actions))]).reshape(-1, 1)
            
#             act_hist = torch.cat((act_hist, torch.tensor(new_actions).cuda()), 1)
#             rew_hist = torch.cat((rew_hist, torch.tensor(new_rewards).cuda()), 1)
            
#             new_model_regrets = discount * np.array([np.max(mus[i]) - mus[i][new_actions[i]] 
#                                                  for i in range(b_size)]).reshape(b_size, 1)
#             model_regrets = torch.cat((model_regrets, torch.tensor(new_model_regrets)), 1)
            
#             del new_actions, new_rewards, action_distribution, new_model_regrets
            
#             if print_mode:
#                 print(f'memory usage for seq_len = {seq_len}')
#                 output_cuda_memory()
            
#             torch.cuda.empty_cache()

    
#         chosen_actions = torch.mean(act_hist, axis = 0)

#         print(f"mu_pair = {mu_pair}:")
#         print(f"avg model regret = {torch.mean(torch.sum(model_regrets, axis = 1)).item()}")
#         if my_strategies:
#             print(f"avg gittins regret = {torch.mean(torch.sum(model_regrets, axis = 1)).item()}")
#             # print(f"avg thompson regret = {np.max(mu_pair) - np.mean(thompson_rewards)}")
#         print(f"------")

#         df = pd.DataFrame({'model_actions' : chosen_actions.cpu().detach().numpy()})
#         if my_strategies:
#             # df['thompson'] = thompson_choices
#             df['gittins'] = gittens_choices
            
#         df.plot(ax = axs[index])
#         axs[index].set_xlabel(f'mu_pair = {mu_pair}')
    
#     plt.show()


