# -*- coding: utf-8 -*-
"""
Created on Mon May 10 16:31:30 2021

@author: maxime
"""
import numpy as np

n_clients_per_class = np.zeros(4) #nombre de clients qui viennent dans la journée par classe, initialisé à zéro. Devra être une variable aléatoire par la suite.

means_per_class = np.array([50, 20, 10, 5])
std_per_class = np.array([10, 10, 1, 3])

for i in range(4):
    x = np.random.normal(means_per_class[i], std_per_class[i])
    if x > 0 : n_clients_per_class[i] = np.round(x)
    else : n_clients_per_class[i] = 0

print(f"\n Aujourd'hui le nombre de client par classe est : {n_clients_per_class} \n ")

p1 = 100; # prix de l'objet 1 
d1 = 50; # coût de production fixe de l'objet 1 

def m1(p1):  # marge faite sur l'objet 1 fonction du prix.
    return p1-d1

def f1_linear(p):
    p_max_per_class = np.array([2*p1, 2.5*p1, 4*p1, 5*p1]) # mouais 
    return np.array([1 - p/p_max for p_max in p_max_per_class]) # il faudrait que ce soit des valeurs entières. pour l'instant je laisse comme ça pour tester un solveur. 

print(f"Pour le prix {p1}$ considéré le taux de conversion par classe est {f1_linear(p1)} \n") 

n_clients_1_per_class = np.round(f1_linear(p1) * n_clients_per_class) # nombre de clients ayant acheté l'item 1 par classe aujourd'hui

n_clients_1 = np.sum(np.round(f1_linear(p1) * n_clients_per_class)) # nombre de clients ayant acheté l'item 1 aujourd'hui  

print(f"Aujourd'hui {n_clients_1} clients ont été conquis par l'article 1, en voici la répartition par classe {n_clients_1_per_class} \nDonc en fait on s'est fait un bénéfice de {n_clients_1 * m1(p1)}$ aujourd'hui, vive la pêche! \n")


alpha = np.array([[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]]) # matrice des promotions, les coordonnées = bails à optimiser. 
                                                                           # initialisé à aucune promotion 
                                                                    
P = np.array([0, 0.10, 0.25, 0.40]) # réductions sur le deuxième article : P0, P1, P2, P3 

p2_initial = 100 # prix initial du deuxième article 

d2 = 50 # coût fixe de production du deuxième article 

p2 = (1-P) * p2_initial # prix proposé en fonction de la réduction.  

m2 = p2 - d2 

print(f"Sur l'achat du second article on se fait une marge de {m2}$ en fonction de la réduction \n")

def f2_linear(p_vector):
    p_max_per_class = np.array([2*p2_initial, 2.5*p2_initial, 4*p2_initial, 5*p2_initial])
    f2 = np.zeros((4,4))
    for j,p in enumerate(p2):
        f2[:,j] = np.array([1 - p/p_max for p_max in p_max_per_class])
    return f2

print(f"Ci-dessous la matrice de taux de conversion avec par ligne les classes et par colonnes les réductions : \n {f2_linear(p2)} \n")

print(f"conversion rate with no promo \n{alpha*f2_linear(p2)} with classes per line and promo per colon") 

def objective_function(p1, p2_initial, alpha):

   return np.dot(f1_linear(p1), n_cients_per_class)*m1(p1)

