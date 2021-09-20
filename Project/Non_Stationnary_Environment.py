# -*- coding: utf-8 -*-
import random
import numpy as np 

from Environment2 import Environment

class Non_Stationnary_Environment(Environment):
    def __init__(self, n_arms,  probabilities_1, probabilities_2, horizon):
        super().__init__(n_arms,  probabilities_1, probabilities_2)
        self.t = 0 
        n_phases = len(self.probabilities_1)
        self.phase_size = horizon/n_phases
        
    def round(self, pulled_arm,n_clients_per_class,new_round = False):
        cv_rate_1 = []
        cv_rate_2 = [[], [], [], []]
        current_phase = int(self.t/ self.phase_size)
        self.t += 1
            
        for i,n in enumerate(n_clients_per_class):
            for k in range(n) : #e
                success_1 = np.random.binomial(1, self.probabilities_1[current_phase][pulled_arm, i])
                cv_rate_1.append(success_1)
                
                if success_1 == 1 :
                    #Pick a promo according to a probability distribution                        
                    proba_promo_per_class = (self.probabilities_2[4*i], self.probabilities_2[4*i+1],self.probabilities_2[4*i+2],self.probabilities_2[4*i+3])       
                    promo_rnd  = random.choices([0,1,2,3], weights=proba_promo_per_class, k=1)[0]
                    success_2 = np.random.binomial(1, self.probabilities_2[4*i+promo_rnd])
                    cv_rate_2[promo_rnd].append(success_2)

        return cv_rate_1, cv_rate_2

    
    