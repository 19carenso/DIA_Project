# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 18:01:15 2021

@author: Fabien
"""
import numpy as np
from hungarian_algorithm import *
from q1 import objective_function

#Pour initialiser la matrice des poids, on a besoin de calculer le profit pour chaque pair classe/promo.
#On met tout le poids d'une promo sur un groupe
#Le problème est que pour calculer le profit on a besoin de fixer les poids des autres classes
#J'ai choisi de répartir uniformément les poids des promos pour les autres classes mais peut-être que ce n'est pas pertinent

adj_mat = np.zeros(16)
n_clients_per_class = [50, 10, 40, 50]
p1 = 50
p2_initial = 50

#1,1
alpha  = [[1, 0, 0, 0],
             [0.25, 0.25, 0.25, 0.25],
             [0.25, 0.25, 0.25, 0.25],
             [0.25, 0.25, 0.25, 0.25]]
adj_mat[0] = -objective_function(p1, p2_initial, alpha, n_clients_per_class)

#1,2
alpha  = [[0, 1, 0, 0],
             [0.25, 0.25, 0.25, 0.25],
             [0.25, 0.25, 0.25, 0.25],
             [0.25, 0.25, 0.25, 0.25]]
adj_mat[1] = -objective_function(p1, p2_initial, alpha, n_clients_per_class)

#1,3
alpha  = [[0, 0, 1, 0],
             [0.25, 0.25, 0.25, 0.25],
             [0.25, 0.25, 0.25, 0.25],
             [0.25, 0.25, 0.25, 0.25]]
adj_mat[2] = -objective_function(p1, p2_initial, alpha, n_clients_per_class)

#1,4
alpha  = [[0, 0, 0, 1],
             [0.25, 0.25, 0.25, 0.25],
             [0.25, 0.25, 0.25, 0.25],
             [0.25, 0.25, 0.25, 0.25]]
adj_mat[3] = -objective_function(p1, p2_initial, alpha, n_clients_per_class)


#2,1
alpha  = [[0.25, 0.25, 0.25, 0.25],
             [1, 0, 0, 0],
             [0.25, 0.25, 0.25, 0.25],
             [0.25, 0.25, 0.25, 0.25]]
adj_mat[4] = -objective_function(p1, p2_initial, alpha, n_clients_per_class)

#2,2
alpha  = [[0.25, 0.25, 0.25, 0.25],
             [0, 1, 0, 0],
             [0.25, 0.25, 0.25, 0.25],
             [0.25, 0.25, 0.25, 0.25]]
adj_mat[5] = -objective_function(p1, p2_initial, alpha, n_clients_per_class)

#2,3
alpha  = [[0.25, 0.25, 0.25, 0.25],
             [0, 0, 1, 0],
             [0.25, 0.25, 0.25, 0.25],
             [0.25, 0.25, 0.25, 0.25]]
adj_mat[6] = -objective_function(p1, p2_initial, alpha, n_clients_per_class)

#2,4
alpha  = [[0.25, 0.25, 0.25, 0.25],
             [0, 0, 0, 1],
             [0.25, 0.25, 0.25, 0.25],
             [0.25, 0.25, 0.25, 0.25]]
adj_mat[7] = -objective_function(p1, p2_initial, alpha, n_clients_per_class)

#3,1
alpha  = [[0.25, 0.25, 0.25, 0.25],
             [0.25, 0.25, 0.25, 0.25],
              [1, 0, 0, 0],
             [0.25, 0.25, 0.25, 0.25]]
adj_mat[8] = -objective_function(p1, p2_initial, alpha, n_clients_per_class)

#3,2
alpha  = [[0.25, 0.25, 0.25, 0.25],
             [0.25, 0.25, 0.25, 0.25],
              [0, 1, 0, 0],
             [0.25, 0.25, 0.25, 0.25]]
adj_mat[9] = -objective_function(p1, p2_initial, alpha, n_clients_per_class)

#3,3
alpha  = [[0.25, 0.25, 0.25, 0.25],
             [0.25, 0.25, 0.25, 0.25],
              [0, 0, 1, 0],
             [0.25, 0.25, 0.25, 0.25]]
adj_mat[10] = -objective_function(p1, p2_initial, alpha, n_clients_per_class)

#3,4
alpha  = [[0.25, 0.25, 0.25, 0.25],
             [0.25, 0.25, 0.25, 0.25],
              [0, 0, 0, 1],
             [0.25, 0.25, 0.25, 0.25]]
adj_mat[11] = -objective_function(p1, p2_initial, alpha, n_clients_per_class)


#4,1
alpha  = [[0.25, 0.25, 0.25, 0.25],
             [0.25, 0.25, 0.25, 0.25],
              [0.25, 0.25, 0.25, 0.25],
            [1, 0, 0, 0]]
adj_mat[12] = -objective_function(p1, p2_initial, alpha, n_clients_per_class)

#4,2
alpha  = [[0.25, 0.25, 0.25, 0.25],
             [0.25, 0.25, 0.25, 0.25],
              [0.25, 0.25, 0.25, 0.25],
            [0, 1, 0, 0]]
adj_mat[13] = -objective_function(p1, p2_initial, alpha, n_clients_per_class)

#4,3
alpha  = [[0.25, 0.25, 0.25, 0.25],
             [0.25, 0.25, 0.25, 0.25],
              [0.25, 0.25, 0.25, 0.25],
            [0, 0, 1, 0]]
adj_mat[14] = -objective_function(p1, p2_initial, alpha, n_clients_per_class)

#4,4
alpha  = [[0.25, 0.25, 0.25, 0.25],
             [0.25, 0.25, 0.25, 0.25],
              [0.25, 0.25, 0.25, 0.25],
            [0, 0, 0, 1]]
adj_mat[15] = -objective_function(p1, p2_initial, alpha, n_clients_per_class)

adj_mat = adj_mat.reshape((4,4))

res = hungarian_algorithm(adj_mat)
print(res[1])
print(adj_mat)
