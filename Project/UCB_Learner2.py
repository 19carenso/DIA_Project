# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 15:48:16 2021

@author: maxime
"""

from Learner2 import Learner
import numpy as np 

class UCB_Learner(Learner):

    def __init__(self, n_arms, p1, p2, alpha, n_clients_per_class):
        '''
        Il est important d'initialiser les valeurs des moyennes avec une 
        valeure supérieure au profit optimum et non avec un vecteur de 0
        Le problème se situe dans l'update. Puisque notre profit a une grande valeur
        et non 0 ou 1, l'upperbound du bras choisi est largement réhaussée à chaque update
        ce qui ammène à tirer toujours le même bras.
        '''        
        super().__init__(n_arms, p1, p2, alpha, n_clients_per_class)    
        self.empirical_means_1 = np.array([0]*n_arms)
        self.confidence_1 = np.array([np.inf]*n_arms)
        
        self.empirical_means_2 = np.array([0]*4)
        self.confidence_2 = np.array([np.inf]*4)
        
    def pull_arm(self): 
        upper_conf = self.empirical_means + self.confidence #### WARNING : self.empircal_mean est mal défini !
        return np.random.choice(np.where(upper_conf == upper_conf.max())[0])
    
    def update(self, pulled_arm, cv_rate_1, cv_rate_2, profit):
        self.t += 1
        self.update_observations(pulled_arm, cv_rate_1, profit)
        self.empirical_means[pulled_arm] = (self.empirical_means[pulled_arm] * (self.t - 1) + profit) / self.t ## updating mean reward of that arm  
        for a in range(self.n_arms):
            n_samples = len(self.cv_rate_1_per_arm[a])
            self.confidence[a] = (2*np.log(self.t)/n_samples)**0.5 if n_samples > 0 else np.inf
        