# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 17:49:41 2021

@author: maxime
"""

import numpy as np 

class Environment():
    def __init__(self, n_arms, probabilities):
        self.n_arms = n_arms
        self.probabilities = probabilities
        
    def round(self, pulled_arm):
        reward = np.random.binomial(1, self.probabilities[pulled_arm])
        return reward