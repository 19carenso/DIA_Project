# -*- coding: utf-8 -*-
"""
Created on Mon May 10 22:39:22 2021

@author: maxime
"""
import numpy as np 

def f1_linear(p1):
    p_max_per_class = np.array([2*p1, 2.5*p1, 4*p1, 5*p1]) # mouais 
    return np.array([1 - p1/p_max for p_max in p_max_per_class])

def m1(p1):  # marge faite sur l'objet 1 fonction du prix.
    d1 = 50; # coût de production fixe de l'objet 1 
    return p1-d1

def m2(p2): # marge faite sur l'objet 2 en fonction du prix avec réduction
    d2 = 50 # coût fixe de production du deuxième article
    return p2 - d2

def f2_linear(p2):
    p2_initial = p2[0]
    p_max_per_class = np.array([2*p2_initial, 2.5*p2_initial, 4*p2_initial, 5*p2_initial])
    f2 = np.zeros((4,4))
    for j,p in enumerate(p2):
        f2[:,j] = np.array([1 - p/p_max for p_max in p_max_per_class])
    return f2

def objective_function(p1, p2_initial, alpha):
   n_clients_per_class = np.zeros(4) #nombre de clients qui viennent dans la journée par classe, initialisé à zéro. Devra être une variable aléatoire par la suite.
   means_per_class = np.array([50, 20, 10, 5])
   std_per_class = np.array([10, 10, 1, 3])
   
   for i in range(4):
       x = np.random.normal(means_per_class[i], std_per_class[i])
       if x > 0 : n_clients_per_class[i] = np.round(x)
       else : n_clients_per_class[i] = 0
           
   n_clients_1_per_class = np.round(f1_linear(p1) * n_clients_per_class) # nombre de clients ayant acheté l'item 1 par classe aujourd'hui

   n_clients_1 = np.sum(np.round(n_clients_1_per_class)) # nombre de clients ayant acheté l'item 1 aujourd'hui 
   
   P = np.array([0, 0.10, 0.25, 0.40]) # Nos promotions, constantes. 
 
   p2 = (1-P) * p2_initial # prix proposé en fonction de la réduction. attention c'est donc un vecteur.
   
   print("allo")
   
   return n_clients_1*m1(p1)  + np.dot(n_clients_1_per_class,np.dot(np.round(alpha*f2_linear(p2)),m2(p2)))