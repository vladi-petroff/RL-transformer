import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt


CONST_ADD = -0.5
# I add -0.5 to original 0/1 rewards to make them zero-mean, this speeds up the training because of reduced variance

class MAB:
    def __init__(self, T = 20, mu_list = None, CONST_ADD = CONST_ADD):
        self.__K = len(mu_list)
        self.__mu_list = mu_list
        self.__T = T
        self.__record = np.zeros((self.__K,2))
        self.__regrets = []
        self.__CONST_ADD = CONST_ADD

    def pull(self, ind):
        reward = 1 * (random.random() < self.__mu_list[ind])
        self.__record[ind, reward] += 1
        self.__regrets.append(max(self.__mu_list) - self.__mu_list[ind])
        return reward + CONST_ADD
        #return (2 * reward - 1) * self.__CONST_ADD

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
    
    def get_CONST_ADD(self):
        return self.__CONST_ADD
