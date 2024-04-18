import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config



class TransformerModelReg(nn.Module):
    '''
    GPT2 transformer with symmetry regulization imposed inside a cyclic group of arm shifts
    forward pass is as in original transofrmer plus returns the "symmetry" variance (which is further used inside the loss function during training)
    '''
    def __init__(self, args, n_dims = 2, max_input_len = 128):
        super(TransformerModelReg, self).__init__()
        configuration = GPT2Config(
            vocab_size = 0, # NEW
            n_positions = max_input_len, # NEW
            
            n_embd = args['n_embd'],
            n_layer = args['n_layer'],
            n_head = args['n_head'],
            use_cache = args['use_cache'],
            
            embd_pdrop = 0.0,
            resid_pdrop = 0.0,
            attn_pdrop = 0.0,
        )
        self.name = f"TransformerModelReg_embd={args['n_embd']}_layer={args['n_layer']}_head={args['n_head']}"
        self.n_arms = args['n_arms']
        self.use_cache = args['use_cache']
        
        self._read_in = nn.Linear(n_dims, args['n_embd'])
        self._backbone = GPT2Model(configuration)
        self._policy_read_out = nn.Linear(args['n_embd'], args['n_arms'])
        self._value_read_out = nn.Linear(args['n_embd'], 1)
        

    @staticmethod
    def _combine(actions_batch, rewards_batch):
        assert (rewards_batch.shape[0] == actions_batch.shape[0]) and (rewards_batch.shape[1] == actions_batch.shape[1])
        
        bsize, seq_len, dim = actions_batch.shape
        together = torch.stack((actions_batch, rewards_batch), axis=2).squeeze(3)
        return together.view(bsize, seq_len, 2 * dim)
    
    
    @staticmethod
    def variance(logprobs_tensors):
        stacked = torch.stack(logprobs_tensors)
        stacked = nn.Softmax(dim=-1)( stacked ) # transforming from logprobs to probs
        stacked = torch.log( stacked )
        
        means = stacked.mean(dim=0)
        squared_diff = (stacked - means)**2
        variance = squared_diff.mean(dim=0)
        return variance
    

    def forward(self, actions, rewards, symmetric_pass = False, past = None):
        assert actions.dtype == torch.long, f"actions.dtype = {actions.dtype}"
        
        if symmetric_pass:
            
            all_actions, all_rewards = [], []
            shifts = np.arange(self.n_arms)
            for SH in shifts:
                actions_shifted = (actions + SH) % self.n_arms
                all_actions.append(actions_shifted)
                all_rewards.append(rewards)
                
            all_actions = torch.cat(all_actions, dim = 0)
            all_rewards = torch.cat(all_rewards, dim = 0)

            zs = self._combine(all_actions, all_rewards)
            embeds = self._read_in(zs.float())

            output = self._backbone(inputs_embeds=embeds).last_hidden_state
            all_policy_out = self._policy_read_out(output)
            all_value_out = self._value_read_out(output)
            separated_policies, separated_values = [], []
            
            mini_b_size = actions.shape[0]
            policy_out, value_out = None, None
            for SH in shifts:
                policy_SH = all_policy_out[SH * mini_b_size : (SH + 1) * mini_b_size]
                
                if policy_out is None:
                    policy_out = torch.roll(policy_SH, shifts = -SH, dims = -1)
                    value_out = all_value_out[SH * mini_b_size : (SH + 1) * mini_b_size]
                else:
                    policy_out += torch.roll(policy_SH, shifts = -SH, dims = -1)
                    value_out += all_value_out[SH * mini_b_size : (SH + 1) * mini_b_size]
                
                separated_policies.append(torch.roll(policy_SH, shifts = -SH, dims = -1))
                separated_values.append(all_value_out[SH * mini_b_size : (SH + 1) * mini_b_size])
                    

            # symmetry_variance = self.variance(separated_policies) + self.variance(separated_values)
            symmetry_variance = self.variance(separated_policies)
            
            return {'policy' : policy_out / len(shifts), 
                    'value' : value_out / len(shifts),
                    'symmetry_variance' : symmetry_variance,
                    'past_key_values' : None }
        
        else:
            
            zs = self._combine(actions, rewards)
            embeds = self._read_in(zs.float())

            output = self._backbone(inputs_embeds=embeds).last_hidden_state
            policy_out = self._policy_read_out(output)
            value_out = self._value_read_out(output)

            return {'policy' : policy_out, 
                    'value' : value_out,
                    'past_key_values' : None }
        
        
        
        

########################
class TransformerModelSymmetric(nn.Module):
    '''
    original GPT2's forward pass is modified by looking at all possible cyclic shifts (imposed on arms) and averaging out results
    '''
    def __init__(self, n_arms, n_dims, n_embd, n_layer, n_head, use_cache, symmetric = False, max_input_len = 128):
        super(TransformerModelSymmetric, self).__init__()
        configuration = GPT2Config(
            vocab_size = 0, # NEW
            n_positions = max_input_len, # NEW
            
            n_embd = n_embd,
            n_layer=n_layer,
            n_head=n_head,
            embd_pdrop=0.0,
            resid_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=use_cache,
        )
        self.name = f"TransformerModelSymmetric_embd={n_embd}_layer={n_layer}_head={n_head}_symmetric={symmetric}"
        self.n_arms = n_arms
        self.use_cache = use_cache
        self.symmetric = symmetric
        print(f"Transformer symmetric = {self.symmetric}, dimensions = {self.name}")
        
        self._read_in = nn.Linear(n_dims, n_embd)
        self._backbone = GPT2Model(configuration)
        self._policy_read_out = nn.Linear(n_embd, n_arms)
        self._value_read_out = nn.Linear(n_embd, 1)
        

    @staticmethod
    def _combine(actions_batch, rewards_batch):
        assert (rewards_batch.shape[0] == actions_batch.shape[0]) and (rewards_batch.shape[1] == actions_batch.shape[1])
        
        bsize, seq_len, dim = actions_batch.shape
        together = torch.stack((actions_batch, rewards_batch), axis=2).squeeze(3)
        return together.view(bsize, seq_len, 2 * dim)


    def forward(self, actions, rewards, past = None):
        assert actions.dtype == torch.long, f"actions.dtype = {actions.dtype}"
        
        if self.symmetric:
            
            all_actions, all_rewards = [], []
            shifts = np.arange(self.n_arms)
            for SH in shifts:
                actions_shifted = (actions + SH) % self.n_arms
                all_actions.append(actions_shifted)
                all_rewards.append(rewards)
                
            all_actions = torch.cat(all_actions, dim = 0)
            all_rewards = torch.cat(all_rewards, dim = 0)

            zs = self._combine(all_actions, all_rewards)
            embeds = self._read_in(zs.float())

            output = self._backbone(inputs_embeds=embeds).last_hidden_state
            all_policy_out = self._policy_read_out(output)
            all_value_out = self._value_read_out(output)
            
            mini_b_size = actions.shape[0]
            policy_out, value_out = None, None
            for SH in shifts:
                policy_SH = all_policy_out[SH * mini_b_size : (SH + 1) * mini_b_size]
                
                if policy_out is None:
                    policy_out = torch.roll(policy_SH, shifts = -SH, dims = -1)
                    value_out = all_value_out[SH * mini_b_size : (SH + 1) * mini_b_size]
                else:
                    policy_out += torch.roll(policy_SH, shifts = -SH, dims = -1)
                    value_out += all_value_out[SH * mini_b_size : (SH + 1) * mini_b_size]
                
                
            return {'policy' : policy_out / len(shifts), 
                    'value' : value_out / len(shifts),
                    'past_key_values' : None }
        
        else:
            
            zs = self._combine(actions, rewards)
            embeds = self._read_in(zs.float())

            output = self._backbone(inputs_embeds=embeds).last_hidden_state
            policy_out = self._policy_read_out(output)
            value_out = self._value_read_out(output)

            return {'policy' : policy_out, 
                    'value' : value_out,
                    'past_key_values' : None }
        
########################