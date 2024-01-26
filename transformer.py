import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config

    

    
# for this model n_dims = 2, meaning that every input is 2-dimensional vector <action, reward>
class TransformerModel2(nn.Module):
    def __init__(self, n_arms, n_dims = 2, n_embd = 16, n_layer = 8, n_head = 4):
        super(TransformerModel2, self).__init__()
        configuration = GPT2Config(
            n_embd = n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        )
        self.name = f"gpt2_embd={n_embd}_layer={n_layer}_head={n_head}"
        self.n_arms = n_arms
        self._read_in = nn.Linear(n_dims, n_embd)
        self._backbone = GPT2Model(configuration)
        self._read_out = nn.Linear(n_embd, n_arms) # retuns logits of arm pull probabilities



    @staticmethod
    def _combine(actions_batch, rewards_batch):
        assert (rewards_batch.shape[0] == actions_batch.shape[0]) and (rewards_batch.shape[1] == actions_batch.shape[1])
        
        bsize, seq_len, dim = actions_batch.shape
        together = torch.stack((actions_batch, rewards_batch), axis=2).squeeze(3)
        return together.view(bsize, seq_len, 2 * dim)


    def forward(self, actions, rewards, inds=None):
        if inds is None:
            inds = torch.arange(actions.shape[1])
        else:
            inds = torch.tensor(inds)
            if max(inds) >= actions.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")


        zs = self._combine(actions, rewards)
        embeds = self._read_in(zs.float())
        output = self._backbone(inputs_embeds=embeds).last_hidden_state
        prediction = self._read_out(output)

        return prediction # return only the very last output
    
    
    
    
    
    
    
    
# class TransformerModel(nn.Module):
#     def __init__(self, n_arms, n_dims = 1, n_embd = 16, n_layer = 8, n_head = 4):
#         super(TransformerModel, self).__init__()
#         configuration = GPT2Config(
#             #n_positions=2 * n_positions,
#             n_embd = n_embd,
#             n_layer=n_layer,
#             n_head=n_head,
#             resid_pdrop=0.0,
#             embd_pdrop=0.0,
#             attn_pdrop=0.0,
#             use_cache=False,
#         )
#         self.name = f"gpt2_embd={n_embd}_layer={n_layer}_head={n_head}"

#         self.n_arms = n_arms
#         self._read_in = nn.Linear(n_dims, n_embd)
#         self._backbone = GPT2Model(configuration)
#         self._read_out = nn.Linear(n_embd, n_arms) # retuns logits of arm pull probabilities



#     @staticmethod
#     def _combine(actions_batch, rewards_batch):
#         """Interleaves the actions and the rewards into a single sequence."""

#         assert (rewards_batch.shape[0] == actions_batch.shape[0]) and (rewards_batch.shape[1] == actions_batch.shape[1])

#         bsize, seq_len, dim = actions_batch.shape
#         together = torch.stack((actions_batch, rewards_batch), axis=2)
#         return together.view(bsize, 2 * seq_len, dim)


#     def forward(self, actions, rewards, inds=None):
#         if inds is None:
#             inds = torch.arange(actions.shape[1])
#         else:
#             inds = torch.tensor(inds)
#             if max(inds) >= actions.shape[1] or min(inds) < 0:
#                 raise ValueError("inds contain indices where xs and ys are not defined")


#         zs = self._combine(actions, rewards)
#         embeds = self._read_in(zs.float())
#         output = self._backbone(inputs_embeds=embeds).last_hidden_state
#         prediction = self._read_out(output)

#         return prediction # return only the very last output