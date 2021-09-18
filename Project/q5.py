# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 18:01:15 2021

@author: Fabien
"""
import numpy as np
from Hungarian_Algorithm import hungarian_algorithm
from q1_functions import profit_per_class


#Attention jusqu'à maintenant on a choisi les valeurs des paramètres un peu au pif ce qui donne un profit par classe très variable selon les classes
#Il semblerait que l'algo hongrois fonctionne mieux quand les classes engendrent un profit similaire
adj_mat = np.zeros(16)
n_clients_per_class = [50,40,25,15]
p1 = 45
p2_initial = 40

  
def assignment(p1, p2_initial, n_clients_per_class) :
    res = (np.zeros(16).reshape(4,4),np.zeros(16).reshape(4,4))
    
    adj_mat = np.zeros(16)   
    n_clients_per_class_rnd = list(np.round(n_clients_per_class + np.random.normal(0,2,4))) #simulation of the number of customer per class according to a normal distribution
    for i in range (0,4):
        #i,1
        alpha  = [1, 0, 0, 0]
        adj_mat[4*i+0] = profit_per_class(i,p1, p2_initial, alpha, n_clients_per_class_rnd)
        #i,2
        alpha  = [0, 1, 0, 0]
        adj_mat[4*i+1] = profit_per_class(i,p1, p2_initial, alpha, n_clients_per_class_rnd)
        #i,3
        alpha  = [0, 0, 1, 0]
        adj_mat[4*i+2] = profit_per_class(i,p1, p2_initial, alpha, n_clients_per_class_rnd)            
        #i,4
        alpha  = [0, 0, 0, 1]                     
        adj_mat[4*i+3] = profit_per_class(i,p1, p2_initial, alpha, n_clients_per_class_rnd)

        
    adj_mat = adj_mat.reshape((4,4))
    res = hungarian_algorithm(np.max(adj_mat) - adj_mat)
    
    return res, adj_mat


res = assignment(p1, p2_initial, n_clients_per_class)
print("\nThe matrix of profit is : ")
print(res[1].round(1))
print("\nThe assignement is : ")
print(res[0])

