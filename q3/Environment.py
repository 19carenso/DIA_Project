# -*- coding: utf-8 -*-
"""
Created on Sun May 16 18:10:05 2021

@author: Fabien
"""
import numpy as np 

class Environment():
    def __init__(self, n_arms, probabilities):
        self.n_arms = n_arms
        self.probabilities = probabilities
        
    def round(self, pulled_arm):
        cv_rate_1 =  np.random.binomial(1, self.probabilities[pulled_arm])
        return cv_rate_1
    
