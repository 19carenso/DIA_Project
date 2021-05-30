# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 15:48:16 2021

@author: maxime
"""

from Learner import Learner
import numpy as np 

class UCB_Learner(Learner):

    def __init__(self, n_arms, p1, p2, alpha, n_clients_per_class):
        super().__init__(n_arms, p1, p2, alpha, n_clients_per_class)    
        self.empirical_means = np.zeros(n_arms)
        self.confidence = np.array([np.inf]*n_arms)
        
    def pull_arm(self): 
        upper_conf = self.empirical_means + self.confidence
        return np.random.choice(np.where(upper_conf == upper_conf.max())[0])
    
    def update(self, pulled_arm, cv_rate_1, profit):
        self.t += 1
        self.update_observations(pulled_arm, cv_rate_1, profit)
        self.empirical_means[pulled_arm] = (self.empirical_means[pulled_arm] * (self.t - 1) + profit) / self.t ## updating mean reward of that arm  
        for a in range(self.n_arms):
            n_samples = len(self.cv_rate_1_per_arm[a])
            self.confidence[a] = (2*np.log(self.t)/n_samples)**0.5 if n_samples > 0 else np.inf
        