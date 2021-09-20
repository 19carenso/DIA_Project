# -*- coding: utf-8 -*-
import numpy as np 

class Environment():
    def __init__(self, n_arms, probabilities_1, probabilities_2):
        self.n_arms = n_arms
        self.probabilities_1 = probabilities_1
        self.probabilities_2 = probabilities_2
                                                
    def round(self, pulled_arm, n_clients_per_class):
        cv_rate_1 = []
        for i,n in enumerate(n_clients_per_class):
            for k in range(n) : #expérienc individuelle pour chaque client de la classe i             
                success_1 = np.random.binomial(1, self.probabilities_1[pulled_arm, i]) # est ce que ce client a acheté l'objet 1?
                cv_rate_1.append(success_1)
        return cv_rate_1, 

