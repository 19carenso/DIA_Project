# -*- coding: utf-8 -*-
"""
Created on Sun May 16 18:16:13 2021

@author: Fabien
"""

from Learner2 import Learner
from q1_functions import objective_function_cv_rates_unknown
import numpy as np

class TS_Learner(Learner):
    def __init__(self, n_arms, P1, p2, alpha, n_clients_per_class):
        super().__init__(n_arms, P1, p2, alpha, n_clients_per_class)
        self.beta_parameters_1 = np.ones((n_arms, 2)) #beta distrib is defined by 2 parameters
        self.beta_parameters_2 = np.ones((4, 2)) # il y a 4 promos et 2 paramètres par promo. 
        
    def pull_arm(self):
        '''The function objective_function needs p1 as an int so we iterate over the possible values of p1
        Each one is multiplied by the conversion rate 1 associated to this price (estimated by sample)
        Then our TS_learner returns the arm (price) to pull which maximize the expected profit
        '''
        sample_1 = np.random.beta(self.beta_parameters_1[:,0], self.beta_parameters_1[:,1]) #this returns an estimation of the conversion rate of item 1
                                                                                          # depending on the pulled arm
                                                                                    
        cv_2_estimation = np.random.beta(self.beta_parameters_2[:,0], self.beta_parameters_2[:,1]) # draw an estimation of cv_2                                                                                   
        value = [0]*self.n_arms
        
        for i in range(self.n_arms):   
            value[i] = objective_function_cv_rates_unknown(self.p1[i], sample_1[i], self.p2, cv_2_estimation, self.alpha, self.n_clients) 
        
        idx = np.argmax(value)
        return idx
        
    def update(self, pulled_arm, cv_rate_1, cv_rate_2, profit):
        self.t += 1
        self.update_observations(pulled_arm, cv_rate_1, cv_rate_2, profit)
        #The first parameter stores how many success we have for each arm while the second parameter contains the number of loss
        self.beta_parameters_1[pulled_arm, 0] = self.beta_parameters_1[pulled_arm, 0] + np.sum(cv_rate_1)
        self.beta_parameters_1[pulled_arm, 1] = self.beta_parameters_1[pulled_arm, 1] + np.sum(1 - np.array(cv_rate_1))
        
        for i in range(4): #iterate over the 4 promos 
            self.beta_parameters_2[i,0] = self.beta_parameters_2[i,0] + np.sum(cv_rate_2[i]) 
            self.beta_parameters_2[i,1] = self.beta_parameters_2[i,1] + np.sum(1 - np.array(cv_rate_2[i])) 
        #print( self.beta_parameters_1, self.beta_parameters_2)
    def reset(self, n_arms):
        self.beta_parameters_1 = np.ones((n_arms, 2))
        self.beta_parameters_2 = np.ones((4, 2)) 
        
        
