# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 18:27:31 2021

@author: Fabien
"""

from TS_Learner2 import TS_Learner
import numpy as np 
 
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
           #Cette condition diffrenciant si t<tau et t>tau est surement inutile
           #pour chaque bras, n_arms correspond au nombre de fois où on a tiré ce bras depuis window_size
           if(self.t<=self.window_size):
               n_samples = np.sum(self.pulled_arms == arm)
           else :
               n_samples = np.sum(self.pulled_arms[-self.window_size:] == arm)
           #cv_rate_1_per_arm stocke le nombre de ventes de l'item 1 à chaque fois que le bras "arm" est tiré
           #cum_rew correspond donc au nombre total d'items 1 vendu depuis window_size
           
           cum_rew = np.sum(self.cv_rate_1_per_arm[arm][-n_samples:]) if n_samples > 0 else 0 
           #Je ne suis pas sur à 100% de mon update des paramètres beta mais le 1er paramètre de beta_1 compte le nombre de succès et le second le nombre d'échecs
           self.beta_parameters_1[arm, 0] =  max(1,cum_rew)
           #n_samples correspond au nombre de fois où on a tiré le bras "arm". Pour raisonner en nombre de succès et d'échecs on multiplie par le nombre total de clients
           #Certes cela ne prend pas en compte la variabilité du nombre de clients à chaque round mais en moyenne on est bon. D'où l'intérêt de mettre un petit écart type, cc Jérémy :) 
           self.beta_parameters_1[arm, 1] = max(1,n_samples*self.n_clients - cum_rew)
           #le max permet c'est pour être comme dans le slides du prof
     
        #j-update beta_2 comme aux questions précedantes vu que cv_rate_2 stationnaire 
       for i in range(4): #iterate over the 4 promos 
           self.beta_parameters_2[i,0] = self.beta_parameters_2[i,0] + np.sum(cv_rate_2[i]) 
           self.beta_parameters_2[i,1] = self.beta_parameters_2[i,1] + np.sum(1 - np.array(cv_rate_2[i])) 
