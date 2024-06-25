import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt


REWARD_SHIFT = -0.5
# I add -0.5 to original 0/1 rewards to make them zero-mean, this speeds up the training because of the reduced variance

class MAB:
    '''
    implements the multi-armed bandit environment
    keeps track of pulled arms & observed outcomes (__record) and regrets (__regrets)
    '''
    def __init__(self, T = 20, mu_list = None, REWARD_SHIFT = REWARD_SHIFT):
        self.__K = len(mu_list)
        self.__mu_list = mu_list
        self.__T = T
        self.__record = np.zeros((self.__K,2))
        self.__regrets = []
        self.__REWARD_SHIFT = REWARD_SHIFT

    def pull(self, ind):
        reward = 1 * (random.random() < self.__mu_list[ind])
        self.__record[ind, reward] += 1
        self.__regrets.append(max(self.__mu_list) - self.__mu_list[ind])
        return reward + REWARD_SHIFT
    
    def reset(self):
        self.__record = np.zeros((self.__K,2))
        self.__regrets = []

    def get_record(self):
        return self.__record

    def get_regrets(self):
        return self.__regrets
        # return np.cumsum(self.__regrets)

    def get_T(self):
        return self.__T

    def get_K(self):
        return self.__K
    
    def get_REWARD_SHIFT(self):
        return self.__REWARD_SHIFT
