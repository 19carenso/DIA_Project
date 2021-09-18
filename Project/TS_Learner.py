# -*- coding: utf-8 -*-
"""
Created on Sun May 16 18:16:13 2021

@author: Fabien
"""

from Learner import Learner
from q1_functions import learner_obj_fun_q3
import numpy as np

class TS_Learner(Learner):
    def __init__(self, n_arms, P1, p2, alpha, n_clients_per_class):
        super().__init__(n_arms, P1, p2, alpha, n_clients_per_class)
        self.beta_parameters= np.ones((n_arms, 2)) #beta distrib is defined by 2 parameters
        
    def pull_arm(self):
        '''The function objective_function needs p1 as an int so we iterate over the possible values of p1
        Each one is multiplied by the conversion rate 1 associated to this price (estimated by sample)
        Then our TS_learner returns the arm (price) to pull which maximize the expected profit
        '''
        sample = np.random.beta(self.beta_parameters[:,0], self.beta_parameters[:,1])
        value = [0]*self.n_arms
        for i in range(self.n_arms):   
            value[i] = learner_obj_fun_q3(self.p1[i], sample[i], self.p2, self.alpha, self.n_clients_per_class)      
        idx = np.argmax(value)
        return idx
        
    def update(self, pulled_arm, cv_rate_1, profit):
        self.t += 1
        self.update_observations(pulled_arm, cv_rate_1, profit)
        #The first parameter stores how many success we have for each arm while the second parameter contains the number of loss
        self.beta_parameters[pulled_arm, 0] = self.beta_parameters[pulled_arm, 0] + np.sum(cv_rate_1)
        self.beta_parameters[pulled_arm, 1] = self.beta_parameters[pulled_arm, 1] + np.sum(1 - np.array(cv_rate_1))

        
