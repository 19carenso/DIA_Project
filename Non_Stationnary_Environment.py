# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 18:21:07 2021

@author: maxime
"""

from Environment import Environment

import numpy as np 

class Non_Stationnary_Environment(Environment):
    def __init__(self, n_arms, probabilities, horizon):
        super().__init__(n_arms, probabilities)
        self.t = 0 
        n_phases = len(self.probabilities)
        self.phase_size = horizon/n_phases
        
    def round(self, pulled_arm):
        current_phase = int(self.t / self.phase_size)
        p = self.probabilities[current_phase][pulled_arm]
        reward = np.random.binomial(1,p)
        self.t += 1
        return reward
    
    