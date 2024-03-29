import numpy as np
import random
import sys
import csv
import pandas as pd
import matplotlib.pyplot as plt
import os


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
from experiment import * 
from transformer import *

    


def MAB_loss(chosen_actions_logs, rewards, static_penalty = 0.0):
    probs = torch.exp(chosen_actions_logs)
    return (-1) * torch.mean( torch.sum(chosen_actions_logs, axis = 1) *\
                             torch.sum(rewards, axis = 1) ) + static_penalty * penalty(probs)


def future_rewards(rewards):
    return torch.sum(rewards, dim = 1, keepdim = True) - torch.cumsum(rewards, dim = 1)


# Loss only with rewards lookin in the future (i.e. at time step t, r_t + ... + r_T)
def MAB_loss_reducedvar(chosen_actions_logs, rewards):
    
    future_rew = future_rewards(rewards)[:, :-1]
    logprobs_dot_rewards = torch.sum(torch.mul(future_rew, chosen_actions_logs), axis = 1)
    probs = torch.exp(chosen_actions_logs)

    return (-1) * torch.mean( logprobs_dot_rewards )



def plot_losses(losses, mean_regrets):
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2)
    fig.set_figwidth(10)
    fig.set_figheight(6 * 2)

    axs[0].plot(pd.Series(losses))
    axs[1].plot(pd.Series(mean_regrets))
    
    
    
def train(model, mini_b_size, N_mini_batches, horizon, gamma = 1,
           LR = 0.001, train_steps = 100, loss_type = 'MAB_loss_reducedvar',
           mus_regime = 'uniform', opt = 'adam',
           model_state_path = None, print_memory = False,
           save_every_iter = None, model_save_path = None, run_experiment_every = None):
    
    n_arms = model.n_arms
    if opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr = LR / N_mini_batches)
    if opt == 'asgd':
        optimizer = torch.optim.ASGD(model.parameters(), lr = LR / N_mini_batches)
    if opt == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr = LR / N_mini_batches)
        
    dim_size = 1
    
    
    if model_state_path is not None:
        assert os.path.exists(model_state_path)
        state = torch.load(model_state_path)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        print(f"Downloaded model from {model_state_path}.")
        

    losses, mean_regrets = [], []

    for t in tqdm(range(train_steps)):
        
        optimizer.zero_grad()

        total_loss = 0
        total_regret = 0
        

        for mini_batch in range(N_mini_batches):

            mus = sample_mus(mini_b_size, n_arms, regime = mus_regime) #arm means

            bandits = [MAB(T = horizon, mu_list = mus[i]) for i in range(mini_b_size)]
            
            # first actions are randomly generated by me
            first_actions = torch.tensor(np.random.choice(n_arms, mini_b_size))
            first_rewards = torch.tensor([bandits[i].pull(first_actions[i]) for i in range(mini_b_size)])

            act_hist = first_actions.view(mini_b_size, -1).float()
            rew_hist = first_rewards.view(mini_b_size, -1).float()
            model_regrets = torch.tensor([]).view(mini_b_size, 0).float()

            final_output = torch.tensor([]).cuda().view(mini_b_size, 0, n_arms).float()
            
            discounted_rewards = first_rewards.view(mini_b_size, -1).float()
            discount = 1
            
            for seq_len in range(bandits[0].get_T()):
                discount *= gamma
                output = model(
                  act_hist.view(mini_b_size, seq_len + 1, dim_size).cuda(),
                  rew_hist.view(mini_b_size, seq_len + 1, dim_size).cuda(),
                )
                
                action_distribution = Categorical(logits = output[:, -1, :].detach())
                new_actions = action_distribution.sample().cpu().detach().numpy().reshape(-1, 1)
                new_rewards = np.array([
                    bandits[i].pull(new_actions[i]) for i in range(len(new_actions))]).reshape(-1, 1) # interact with bandits
                

                act_hist = torch.cat((act_hist, torch.tensor(new_actions)), 1).detach()
                rew_hist = torch.cat((rew_hist, torch.tensor(new_rewards)), 1).detach()
                discounted_rewards = torch.cat((discounted_rewards, discount * torch.tensor(new_rewards)), 1).detach()
                
                
                new_model_regrets = discount * np.array([np.max(mus[i]) - mus[i][new_actions[i]] 
                                                 for i in range(mini_b_size)]).reshape(mini_b_size, 1)
                model_regrets = torch.cat((model_regrets, torch.tensor(new_model_regrets)), 1)
                
                
                del new_actions, new_rewards, new_model_regrets, action_distribution

                if seq_len == bandits[0].get_T() - 1:
                    final_output = output
                    del output
                else:
                    del output

                torch.cuda.empty_cache()

                if print_memory: 
                    #helps to see how large can each mini batch be 
                    print(f'memory usage for seq_len = {seq_len}')
                    output_cuda_memory()
            
            
            # assert final_output.size(1) in [bandits[0].get_T(), 2 * bandits[0].get_T()]
            
            # if we use one-dimensional input (i.e. every input is action or reward),
            # then I need to take only every second coordinate
            if final_output.size(1) == 2 * bandits[0].get_T():
                final_output = final_output[:, 1::2, :]
                
            
            log_probs = nn.LogSoftmax(dim = 2)(final_output)            
            idx = (act_hist[:, 1:]).long().view(mini_b_size, -1, 1).cuda().detach()
            selected_log_probs = torch.gather(log_probs, 2, idx).squeeze(2)
            
            if loss_type == 'MAB_loss':
                loss = MAB_loss(selected_log_probs, discounted_rewards.cuda())
            if loss_type == 'MAB_loss_reducedvar':
                loss = MAB_loss_reducedvar(selected_log_probs, discounted_rewards.cuda())
                

            loss.backward()
            total_loss += loss.detach().item()
            total_regret += torch.mean(torch.sum(model_regrets, axis = 1)).item()

            del final_output, log_probs, idx, act_hist, rew_hist, loss
    
            torch.cuda.empty_cache()
        
        
        # to see whether we reached the extremum point (i.e. gradient = 0)
        if run_experiment_every is not None and (t + 1) % run_experiment_every == 0:
            print('\nGRADIENTS:')
            print(f"model._read_in.weight.grad = {model._read_in.weight.grad}")
            print(f"model._read_in.bias.grad = {model._read_in.bias.grad}")
            print(f"model._read_out.weight.grad = {model._read_out.weight.grad}")
            print(f"model._read_out.bias.grad = {model._read_out.bias.grad}")
            print('<------>')

                
        optimizer.step()
        optimizer.zero_grad()
        

        losses.append(total_loss / N_mini_batches)
        mean_regrets.append(total_regret / N_mini_batches)
        
        print(f"cur loss = {losses[-1]}")
        print(f"cur regret = {mean_regrets[-1]}")


        if save_every_iter is not None and model_save_path is not None:
            if (t + 1) % save_every_iter == 0:    
                training_state = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict()}            
                torch.save(training_state, model_save_path)
                print(f'saved model to {model_save_path}')
                
        
        # plots actions of current model state
        # helps to see whether the model training is "moving" in the right direction
        if run_experiment_every is not None and (t + 1) % run_experiment_every == 0:
            run_experiment_upd(model, horizon = horizon, gamma = gamma, 
                               b_size = 100, mu_list = [[0, 1], [1, 0], [0.4, 0.6], [0.6, 0.4]])

        torch.cuda.empty_cache()

    return losses, mean_regrets














# ## penalties & hacks    
# def penalty(probs, cutoff = 0.95):
#     return torch.mean(
#         torch.sum(torch.nn.functional.relu(probs - cutoff), axis = 1) )


# def smoothness_penalty(logprobs):
#     diff = torch.diff(logprobs, dim = 1)
#     return torch.mean(diff.square())


# def smoothness_penalty2(logratios):
#     diff = torch.diff(logratios, dim = 1)
#     shift1 = torch.roll(diff, -1, 1)
#     neighbor_products = diff[:, :-1] * shift1[:, :-1]
#     relu_products = torch.nn.functional.relu((-1) * neighbor_products)
    
#     return torch.mean(torch.sqrt(relu_products))


# def smoothness_penalty3(logratios):
#     diff = torch.diff(logratios, dim = 1)
#     diff_abs = torch.abs(diff)
    
#     return torch.mean(torch.sum(torch.nn.functional.relu(diff_abs - 3), axis = 1))


# def estimate_baselines(future_rewards):
#     return torch.mean(future_rewards, axis = 0)



# def analyze_advantages(act_hist, rewards):
#     act_hist = act_hist[:, 1:]
#     horizon = act_hist.size(1)
    
#     future_rew = future_rewards(rewards)[:, :-1]
#     future_rew_minus_baseline = future_rew - estimate_baselines(future_rew)
    
#     advantage0, advantage1 = [], []
#     for t in range(horizon):
#         action0 = act_hist[:, t] == 0
#         action1 = act_hist[:, t] == 1
        
#         advantage0.append( torch.mean(future_rew_minus_baseline[action0][:, t]).item() )
#         advantage1.append( torch.mean(future_rew_minus_baseline[action1][:, t]).item() )
        
#     df = pd.DataFrame({'adv0' : advantage0, 'adv1' : advantage1})
#     df.plot()
#     plt.show()