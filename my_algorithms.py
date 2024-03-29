import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt

from functools import lru_cache
from env_MAB import *





def random_argmax(a):
    return np.random.choice(np.where(a == a.max())[0])


def sample_mus(b_size, n_arms, regime = 'uniform', offset = 0.0):

    if regime == 'uniform':
        return np.random.rand(b_size, n_arms).astype(float)

    if regime == 'uniform_separated':
        random_picks = np.random.randint(low = 0, high = 2, size = b_size)
        mus = np.array([[0.0, 0.0] for i in range(b_size)])
        for i in range(b_size):
            mus[i] = np.random.rand(2).astype(float) * 0.3
            mus[i][random_picks[i]] += 0.7
        return mus


    if regime == 'random_offset':
        random_picks = np.random.randint(low = 0, high = 2, size = b_size)
        mus = np.array([[offset, offset] for i in range(b_size)])
        for i in range(b_size):
            mus[i][random_picks[i]] = 1 - offset
        return mus


    if regime == 'constant_offset':
        return np.array([[offset, 1 - offset]] * b_size).astype(float)

    assert False, f"incorrect regime = {regime}"



class Thompson_sampling():
    def __init__(self, MAB):
        self.MAB = MAB
        self.K = self.MAB.get_K()
        self.params = {'alpha' : np.ones(self.K), 'beta' : np.ones(self.K)}


    def reset(self):
        self.MAB.reset()
        self.params['alpha'] = np.ones(self.K)
        self.params['beta'] = np.ones(self.K)


    def choose_thompson_arm(self):
        cur_record = self.MAB.get_record()
        beta_draws = [np.random.beta(1 + cur_record[k][1], 1 + cur_record[k][0]) for k in range(self.K)]
        arm = random_argmax(np.array(beta_draws))

        return arm

def estimate_thompson_probs(bandit):
    total_sum = 0
    trials = 50
    cur_record = bandit.get_record()
    for _ in range(trials):
        beta_draws = [np.random.beta(1 + cur_record[arm_k][1], 1 + cur_record[arm_k][0]) for arm_k in range(bandit.get_K())]
        total_sum += random_argmax(np.array(beta_draws))

    return total_sum / trials




#############




class Gittins_index():
    def __init__(self, MAB, gamma=0.90, epsilon=1e-4, N=100):
        self.MAB = MAB

        # Use indepdent uniform priors
        self.alpha = np.ones(MAB.get_K())
        self.beta = np.ones(MAB.get_K())
        self.gamma = gamma
        self.epsilon=epsilon
        self.N = N
        self.lower_bound = 0
        self.upper_bound = 1/(1-self.gamma)
        self.gittins_indices = np.zeros(MAB.get_K())

        self.compute_gittins_index(0)

        for i in range(MAB.get_K()):
            self.gittins_indices[i] = self.gittins_indices[0]
        


    def reset(self):
        '''
        Reset the instance and eliminate history.
        '''
        self.MAB.reset()
        self.alpha = np.ones(self.MAB.get_K())
        self.beta = np.ones(self.MAB.get_K())
        self.lower_bound = 0
        self.upper_bound = 1/(1-self.gamma)
        self.gittins_indices = np.zeros(self.MAB.get_K())

        self.compute_gittins_index(0)

        for i in range(self.MAB.get_K()):
            self.gittins_indices[i] = self.gittins_indices[0]
    
    @lru_cache(maxsize=100) # (used to be lru_cache(maxsize=None))
    def calculate_value_oab(self, successes, total_num_samples, lambda_hat, stage_num=0):
        mean_beta_posterior = successes/total_num_samples
        oab_value = 0
        if stage_num == self.N:
            coefficient = (self.gamma**stage_num)/(1-self.gamma)
            oab_value = coefficient * max(0, (mean_beta_posterior-lambda_hat))
        else:
            immediate_reward = mean_beta_posterior - lambda_hat
            future_success_value = self.calculate_value_oab(successes+1, total_num_samples+1, lambda_hat, stage_num+1)
            future_failure_value = self.calculate_value_oab(successes, total_num_samples+1, lambda_hat, stage_num+1)
            expected_future_reward = self.gamma * ((mean_beta_posterior*future_success_value)+((1-mean_beta_posterior)*future_failure_value))
            oab_value = max(0, (immediate_reward+expected_future_reward))

        return oab_value
    
    def compute_gittins_index(self, arm_index):
        '''
        Calibration for Gittins Index
        '''
        records = self.MAB.get_record()
        wins = records[arm_index, 1]
        losses = records[arm_index, 0]

        # We include a 1 as the first term to account for normal priors
        self.alpha[arm_index] = 1 + wins
        self.beta[arm_index] = 1 + losses

        lambda_hat = 0
        self.lower_bound = 0
        self.upper_bound = 1/(1-self.gamma)
        while self.upper_bound - self.lower_bound > self.epsilon:
            lambda_hat = (self.lower_bound + self.upper_bound)/2
            succeses = self.alpha[arm_index]
            total_num_samples = self.alpha[arm_index] + self.beta[arm_index]

            oab_value = self.calculate_value_oab(succeses, total_num_samples, lambda_hat)
            if oab_value > 0:
                self.lower_bound = lambda_hat
            else:
                self.upper_bound = lambda_hat
        
        self.gittins_indices[arm_index] = lambda_hat


    def play_one_step(self):
        '''
        Select the arm with the highest Gittins Index and about its Gittins Index based on the value return by pull
        '''

        # Select the arm with the largest Gittins Index
        selected_arm = random_argmax(self.gittins_indices)
        self.MAB.pull(selected_arm)

        self.compute_gittins_index(selected_arm)
        return selected_arm
    
    
    def choose_gittins_arm(self):
        # just selects gittins index without actually pulling the arm or updating the value
        selected_arm = random_argmax(self.gittins_indices)
        return selected_arm
    
    
    
    
    
    

    
    
    
# class Gittins_index():
#     def __init__(self, MAB, gamma = 0.5, epsilon = 1e-5, N = None):
#         self.MAB = MAB

#         self.alpha = np.ones(MAB.get_K())
#         self.beta = np.ones(MAB.get_K())
#         self.gamma = gamma
#         self.epsilon=epsilon
#         self.N = N
#         self.lower_bound = 0
#         if self.gamma < 1:
#             self.upper_bound = 1/(1-self.gamma)
#         else:
#             self.upper_bound = self.N

#         self.gittins_indices = np.zeros(MAB.get_K())

#         self.compute_gittins_index(0)

#         for i in range(MAB.get_K()):
#             self.gittins_indices[i] = self.gittins_indices[0]



#     def reset(self):
#         self.MAB.reset()
#         self.alpha = np.ones(self.MAB.get_K())
#         self.beta = np.ones(self.MAB.get_K())
#         self.lower_bound = 0
#         if self.gamma < 1:
#             self.upper_bound = 1/(1-self.gamma)
#         else:
#             self.upper_bound = self.N

#         self.gittins_indices = np.zeros(self.MAB.get_K())

#         self.compute_gittins_index(0)

#         for i in range(self.MAB.get_K()):
#             self.gittins_indices[i] = self.gittins_indices[0]


#     @lru_cache(maxsize=None)
#     def calculate_value_oab(self, successes, total_num_samples, lambda_hat, stage_num=0):
#         mean_reward = successes / total_num_samples
#         if stage_num == self.N:
#             if self.gamma < 1:
#                 coeff = pow(self.gamma, self.N) / (1 - self.gamma)
#             else:
#                 coeff = 1
            
#             return coeff * max(mean_reward - lambda_hat, 0)
#         else:
#             future_reward = mean_reward * self.calculate_value_oab(successes + 1, total_num_samples + 1, lambda_hat, stage_num + 1) +\
#             (1 - mean_reward) * self.calculate_value_oab(successes, total_num_samples + 1, lambda_hat, stage_num + 1)

#             return max(mean_reward - lambda_hat + self.gamma * future_reward, 0)


#     def compute_gittins_index(self, arm_index):
#         arm_record = self.MAB.get_record()[arm_index]

#         self.alpha[arm_index] = 1 + arm_record[1]
#         self.beta[arm_index] = 1 + arm_record[0]

#         lambda_hat = 0
#         self.lower_bound = 0
#         if self.gamma < 1:
#             self.upper_bound = 1/(1-self.gamma)
#         else:
#             self.upper_bound = self.N

#         while self.upper_bound - self.lower_bound > self.epsilon:
#             lambda_hat = (self.lower_bound + self.upper_bound) / 2
#             succeses = self.alpha[arm_index]
#             total_num_samples = self.alpha[arm_index] + self.beta[arm_index]

#             oab_value = self.calculate_value_oab(succeses, total_num_samples, lambda_hat, stage_num = 0)
#             if oab_value > 0:
#                 self.lower_bound = lambda_hat
#             else:
#                 self.upper_bound = lambda_hat

#         self.gittins_indices[arm_index] = lambda_hat


#     def choose_gittins_arm(self):
#         # Select the arm with the largest Gittins Index
#         # print(self.gittins_indices)        
#         selected_arm = random_argmax(self.gittins_indices)
#         self.compute_gittins_index(selected_arm)
#         return selected_arm
    
    
#     def play_one_step(self):
#         selected_arm = random_argmax(self.gittins_indices)
#         self.MAB.pull(selected_arm)
#         self.compute_gittins_index(selected_arm)
#         return selected_arm
