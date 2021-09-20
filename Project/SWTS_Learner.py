# -*- coding: utf-8 -*-
import numpy as np 

from TS_Learner2 import TS_Learner

class SWTS_Learner(TS_Learner):
   def __init__(self,n_arms, P1, p2, alpha, n_clients_per_class, window_size):
        super().__init__( n_arms, P1, p2, alpha, n_clients_per_class)
        self.window_size = window_size
        self.pulled_arms = np.array([])
        
   def update(self, pulled_arm, cv_rate_1, cv_rate_2, profit):
       self.t += 1 
       self.update_observations(pulled_arm, cv_rate_1, cv_rate_2, profit)
       self.pulled_arms = np.append(self.pulled_arms, pulled_arm)
       for arm in range (self.n_arms):
           #for each arm, n_arms corresponds to the number of times the arm has been pulled since window_size
           if(self.t<=self.window_size):
               n_samples = np.sum(self.pulled_arms == arm)
           else :
               n_samples = np.sum(self.pulled_arms[-self.window_size:] == arm)
        
           #cv_rate_1_per_arm stores the number of item 1 sold when each arm is pulled
           #so cum_rew corresponds to the total number of item 1 sold since window_size
           cum_rew = np.sum(self.cv_rate_1_per_arm[arm][-n_samples:]) if n_samples > 0 else 0 
           self.beta_parameters_1[arm, 0] =  max(1,cum_rew)
           self.beta_parameters_1[arm, 1] = max(1,n_samples*self.n_clients - cum_rew)
     
       for i in range(4): #iterate over the 4 promos 
           self.beta_parameters_2[i,0] = self.beta_parameters_2[i,0] + np.sum(cv_rate_2[i]) 
           self.beta_parameters_2[i,1] = self.beta_parameters_2[i,1] + np.sum(1 - np.array(cv_rate_2[i])) 
