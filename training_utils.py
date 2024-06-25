import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from env_MAB import *


### Learning Rate Scheduling
def warmup_coeff(start : float, finish : float, total_len : int, progress_pct :float = 1.0, from_head = True):
    """
    creates a warm-up scheduling: coefficients changes from value = start to value = finish over the course of 
    progress_pct * total_len steps, and then becomes constant. 

    Args:
        start (float): start value of the coefficient
        finish (float): final value of the coefficient
        total_len (int): output list length
        progress_pct (float): how much of the list share will be spent on gradual changes vs. being constant
        from_head (bool): whether to keep initial coefficients constant or final coefficients constant (defaults to final)

    Returns:
        list of floats: multipliers for our original learning rate
    """

    progress_len = int(progress_pct * total_len)
    
    answer = []
    if progress_len > 0:
        step = (finish - start) / progress_len
        answer = [start + i * step for i in range(progress_len)]
    
    if from_head:
        answer += [finish] * (total_len - progress_len)
    else:
        answer = [start] * (total_len - progress_len) + answer
        
    return answer


def create_simple_scheduler(optimizer, warmup_steps):
    """
    creates a simple warm-up scheduler: increase from 1e-2 to 1 over the course of warmup_steps, 
    and always return 1 afterwardse (this is our lambda multiplier depending on epoch)

    Args:
        optimizer (optimizer): torch.optim class instance (we wrap LambdaLR over it)
        warmup_steps (int): for how many steps perform warmup
    Returns:
        torch.optim.lr_scheduler.LambdaLR class instance
    """
        
    warmup = np.interp(np.arange(1 + warmup_steps), [0, warmup_steps], [1e-2, 1])
    lr_lambda = lambda epoch: 1 if epoch >= len(warmup) else warmup[epoch]
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    return scheduler


def create_const_scheduler(optimizer, multiplier):
    lr_lambda = lambda epoch: multiplier
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler



### Loss computation  
class BanditLossComputer():
    '''
    Our final loss function is an complicated combination of several components, 
    so it's easier to incapsulate all of them in one single Loss class
    We later can access individual loss components simply by calling BanditLossComputer.policy_loss_GAE() and etc. in the training loop  
    '''
    
    def __init__(self, args : dict):
        self.gamma = args['gamma']
        self.horizon = args['horizon']
        
    
    def future_rewards(self, rewards):
        '''
        returns the future rewards, a well-known optimization in the REINFORCE method:
        future_rewards[i] = rewards[i] + rewards[i + 1] + ...
        '''
        future_rew = torch.clone(rewards)
        for t in range(rewards.shape[1] - 2, -1, -1):
            future_rew[:, t] += future_rew[:, t + 1]

        return future_rew
    


    def policy_loss_GAE(self, chosen_actions_logs, rewards, values, lambdaa): 
        '''
        REINFORCE policy loss with Generalized Advantage Estimation

        Parameters:
            chosen_actions_logs (torch.Tensor) : log probabilities of actions chosen in these rollouts
            rewards (torch.Tensor) : observed rewards
            values (torch.Tensor) : value network baselines estimates
            lambdaa (float) : controls the bias-variance trade-off in the Generalized Advantage definition 
            (lambdaa = 1 means unbiased gradient but higher variance, and lambdaa = 0 means very greedy biased gradient estimates)

        Returns:
            single float loss (torch.Tensor)
        '''

        assert rewards.shape[1] == values.shape[1]

        values_shift = torch.roll(values, -1, dims = 1)
        values_shift[:, -1] = 0
        deltas = rewards + self.gamma * values_shift - values

        GAE = deltas
        for t in range(deltas.shape[1] - 2, -1, -1):
            GAE[:, t] += self.gamma * lambdaa * GAE[:, t + 1]


        mini_batch_sz, _ = values.shape
        normalizer = self.gamma * torch.vander(torch.tensor([ self.gamma ] * mini_batch_sz), 
                                 N = self.horizon, increasing = True).to( GAE.get_device() )
        GAE = (GAE * normalizer).float()
        
        logprobs_dot_advantages = torch.sum(torch.mul(chosen_actions_logs, GAE), axis = 1)

        return (-1) * torch.mean( logprobs_dot_advantages )



    def entropy_loss(self, policy_output) -> float:
        '''
        entropy regulization loss

        Parameters:
            policy_output: torch.Tensor
                log probabilities (for taking all possible actions) across the full episode

        Returns:
            torch.Tensor with single float loss value

        '''
        entropies = torch.sum(F.softmax(policy_output, dim = 2) * F.log_softmax(policy_output, dim = 2), axis = 2)

        mini_batch_sz = policy_output.shape[0]
        # we use normalizer because our problem discounts rewards in the larter horizons, which affects policy loss accordingly, 
        # and hence we want entropy loss to act likewise (otherwise this loss will dominate later horizons)

        normalizer = self.gamma * torch.vander(torch.tensor([self.gamma] * mini_batch_sz), 
                                  N = self.horizon, increasing = True).to( entropies.get_device() )
        discounted_entropies = (entropies * normalizer).float()

        return torch.mean(torch.sum(discounted_entropies, axis = 1))


    
    def value_loss(self, value_output, disc_rewards):
        '''
        computes the values loss as MSE between value_outputs and observed rewards
        "ideal" loss output would be V_t = E(r_t + gamma * r_{t + 1} + ... )

        Note: we want the value to predict well on every horizon, but the final loss will be discounted
        in a way simiilar to the policy and entropy losses
        '''
        disc_future_rew = self.future_rewards(disc_rewards).float()
        
        mini_batch_sz, _ = disc_future_rew.shape
        
        normalizer = self.gamma * torch.vander(torch.tensor([self.gamma] * mini_batch_sz), 
                                  N = self.horizon, increasing = True).to( value_output.get_device() )
        disc_value_output = (value_output.squeeze(2) * normalizer).float()

        diff = torch.square(disc_value_output - disc_future_rew)

        return torch.mean(torch.sum(diff, axis = 1))
    
    
    
    
    # this loss was used in some experiments to observe whether it goes down with training, especially as we start to approach
    # not actually used for backpropagation
    def invariance_loss(self, policy_output, act_hist, rew_hist):
        '''
        Gittins index (optimal strategy) should be permutation-invariant, 
        i.e., optimal actions should not change if we just shuffle history before the current moment

        this function can be used to calculate invariance loss (which will be zero in case the policy is indeed permutation invariant)
        not used during the training in the final work, but was used for exploration of model's behaviour 
        '''
        assert policy_output.shape[1] == self.horizon and act_hist.shape[1] == self.horizon + 1
        
        mini_b_size = policy_output.shape[0]

        arms_history = {i : {'arm0' : [0, 0], 'arm1' : [0, 0]} for i in range(mini_b_size)}
        def update_history(index, arm, reward):
            outcome = int(reward > 0)
            arms_history[index][f'arm{int(arm)}'][outcome] += 1
        
        
        for i in range(mini_b_size):
            time0 = 0
            update_history(i, act_hist[i][time0].item(), rew_hist[i][time0].item())
            

        loss = 0
        discount = 1
        for t in range(self.horizon):
            discount *= self.gamma

            groupings = {}
            for i in range(mini_b_size):
                update_history(i, act_hist[i][t + 1].item(), rew_hist[i][t + 1].item())
                key = f"{arms_history[i]['arm0']}_{arms_history[i]['arm1']}"
                groupings[key] = []

            for i in range(mini_b_size):
                key = f"{arms_history[i]['arm0']}_{arms_history[i]['arm1']}"
                groupings[key].append(i)


            for key, group in groupings.items():
                if len(group) > 1:
                    invariant_group = policy_output[group][:, t, :]      
                    loss += discount * torch.sum( invariant_group.shape[0] *\
                        torch.var(invariant_group, dim = 0, correction = 0) 
                    )

            discount *= self.gamma

        return loss / mini_b_size
    
    
    
### Other  

def run_experiment(model, horizon, gamma, b_size, dim_size = 1,
                        model_decision = 'sample'):
    
    '''
    runs experiment during the training: calculate the mean regret (minus reward) for "trivial" bandits, where one of the arms has mean = 1, and all others = 0

    this way we can emperically track whether the model favors some arms over others and reveal any asymmetric behaviour
    '''

    n_arms = model.n_arms
    
    print('running experiment:')
    for idx in range(n_arms):
        
        mu_pair = [0] * n_arms
        mu_pair[idx] = 1

        mus = np.array([mu_pair] * b_size).astype(float)
        bandits = [MAB(T = horizon, mu_list = mus[i]) for i in range(b_size)]
        model_regrets = torch.tensor([]).view(b_size, 0).float()        
        
        first_actions = torch.tensor(np.random.choice(n_arms, b_size)).cuda()
        first_rewards = torch.tensor([bandits[i].pull(first_actions[i].item()) for i in range(b_size)]).cuda()
        
        act_hist = first_actions.view(b_size, -1).long()
        rew_hist = first_rewards.view(b_size, -1).float()
        discounted_rewards = first_rewards.view(b_size, -1).float()

        discount = 1
        for seq_len in range(bandits[0].get_T()):
            discount *= gamma
            
            output_dict = model(
                act_hist.view(b_size, seq_len + 1, dim_size),
                rew_hist.view(b_size, seq_len + 1, dim_size),
            )
            policy_output = output_dict['policy']
            
            
            if model_decision == 'sample':
                action_distribution = Categorical(logits = policy_output[:, -1, :].detach())
                new_actions = action_distribution.sample().cpu().detach().numpy().reshape(-1, 1)
            elif model_decision == 'argmax':
                new_actions = torch.argmax(policy_output[:, -1, :].detach(), dim = 1).cpu().detach().numpy().reshape(-1, 1)
                
            new_rewards = np.array([bandits[i].pull(new_actions[i][0]) for i in range(len(new_actions))]).reshape(-1, 1)

            act_hist = torch.cat((act_hist, torch.tensor(new_actions).cuda()), 1)
            rew_hist = torch.cat((rew_hist, torch.tensor(new_rewards).cuda()), 1)
            discounted_rewards = torch.cat((discounted_rewards, discount * torch.tensor(new_rewards).cuda()), 1).detach()
            
            new_model_regrets = discount * np.array([np.max(mus[i]) - mus[i][new_actions[i][0]] 
                                                 for i in range(b_size)]).reshape(b_size, 1)
            model_regrets = torch.cat((model_regrets, torch.tensor(new_model_regrets)), 1)
    
        chosen_actions = torch.mean(act_hist.float(), axis = 0)
        
        print(f"mus = {mu_pair}, avg regret = {torch.mean(torch.sum(model_regrets, axis = 1)).item()}")
        
    print('\n')


## old utils 

def create_cosine_scheduler_upd(optimizer, train_steps, warmup_steps):
    
    warmup = np.interp(np.arange(1 + warmup_steps), [0, warmup_steps], [1e-2, 1])
    normal_steps = train_steps - warmup_steps
    xx = np.arange(normal_steps) / normal_steps
    cosine = (np.cos(np.pi * xx) + 1) / 2
    lr_schedule = np.concatenate([warmup, cosine])
    lr_lambda = lambda epoch: 1 if epoch >= len(lr_schedule) else lr_schedule[epoch]
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    return scheduler
