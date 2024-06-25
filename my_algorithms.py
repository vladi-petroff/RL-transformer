import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from functools import lru_cache
from env_MAB import *


def random_argmax(a):
    return np.random.choice(np.where(a == a.max())[0])


def sample_mus(b_size, n_arms, regime = 'uniform', offset = 0.0):
    '''
    different distributions of mus are used dependin on the training regime
    '''
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
    
    
    if regime == 'beta':
        return np.random.beta(a = 10, b = 10, size = 2 * b_size).reshape(b_size, 2)
    

    assert False, f"incorrect regime = {regime}"


#############
# Thompson Sampling -> used as 'baseline'
# Gittins Index (both with Uniform and Beta preiors -> optimal solution
#############

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
        if self.gamma < 1:
            self.upper_bound = 1/(1-self.gamma)
        else:
            self.upper_bound = self.N
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
    
    @lru_cache(maxsize=200) # (used to be lru_cache(maxsize=None))
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
    
    
    
    

class Gittins_index_beta():
    '''Gittins index with beta priors'''
    def __init__(self, MAB, gamma=0.90, epsilon=1e-4, N=100):
        self.MAB = MAB

        # Use indepdent uniform priors
        self.alpha = np.ones(MAB.get_K())
        self.beta = np.ones(MAB.get_K())
        self.gamma = gamma
        self.epsilon=epsilon
        self.N = N
        self.lower_bound = 0
        if self.gamma < 1:
            self.upper_bound = 1/(1-self.gamma)
        else:
            self.upper_bound = self.N
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
    
    @lru_cache(maxsize=200) # (used to be lru_cache(maxsize=None))
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

        # We include 10 as the first term to account for beta(10, 10) priors
        self.alpha[arm_index] = 10 + wins
        self.beta[arm_index] = 10 + losses

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