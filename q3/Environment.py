# -*- coding: utf-8 -*-
"""
Created on Sun May 16 18:10:05 2021

@author: Fabien
"""
import numpy as np 

class Environment():
    def __init__(self, n_arms, probabilities_1, probabilities_2):
        self.n_arms = n_arms
        self.probabilities_1 = probabilities_1 #Les 4 taux de conversion par classe pour chaque bras
        self.probabilities_2 = probabilities_2 #Les 16 taux de convesion par classe pour chaque promo, les uns après les autres dans une suite
                                               #rangés par promo puis par classe, typiquement les 4 premiers correspondent aux 4 promos pour 
                                               #la classe 1
                                                
    def round(self, pulled_arm, n_clients_per_class):
        cv_rate_1 = []
        '''
        L'environnement a besoin de dissocier les classes des clients pour l'item 1 ET pour l'item 2.
        
        '''
        for i,n in enumerate(n_clients_per_class):
            for k in range(n) : #expérienc individuelle pour chaque client de la classe i             
                success_1 = np.random.binomial(1, self.probabilities_1[pulled_arm, i]) # est ce que ce client a acheté l'objet 1?
                cv_rate_1.append(success_1)
        return cv_rate_1, 

    
    '''
    Vu qu'on utilise une loi beta qui prend en compte le nombre total de succès donc des valeurs entières on ne vas pas pouvoir 
    considérer des valeurs non entières de clients lors de l'association des promos. C'est le cas içi vu que l'environnement retourne
    des valeurs réalistes donc entières mais dans l'estimation faite par le learner ce n'est pas un problème. 
    '''
