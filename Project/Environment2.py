# -*- coding: utf-8 -*-
import random
import numpy as np 

class Environment():
    def __init__(self, n_arms, probabilities_1, probabilities_2):
        self.n_arms = n_arms
        self.probabilities_1 = probabilities_1 
        self.probabilities_2 = probabilities_2 
                                                
    def round(self, pulled_arm, n_clients_per_class):
        cv_rate_1 = []
        cv_rate_2 = [[], [], [], []]

        for i,n in enumerate(n_clients_per_class):
            for k in range(n) :    
                success_1 = np.random.binomial(1, self.probabilities_1[pulled_arm, i])
                cv_rate_1.append(success_1)
                if success_1 == 1 :
                    #Pick a promo according to a probability distribution                        
                    proba_promo_per_class = (self.probabilities_2[4*i], self.probabilities_2[4*i+1],self.probabilities_2[4*i+2],self.probabilities_2[4*i+3])       
                    promo_rnd  = random.choices([0,1,2,3], weights=proba_promo_per_class, k=1)[0]
                    success_2 = np.random.binomial(1, self.probabilities_2[4*i+promo_rnd])
                    cv_rate_2[promo_rnd].append(success_2)
        return cv_rate_1, cv_rate_2
