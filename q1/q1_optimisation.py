# -*- coding: utf-8 -*-
"""
Created on Wed May 12 10:28:24 2021

@author: maxime
"""
import autograd.numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as so
from autograd import grad, jacobian
from q1 import objective_function, margin1, margin2, conversion1, conversion2, test1

def x_constr(p1, p2_initial, alpha):
    x = np.zeros(18)  
    x[0] = p1
    x[1] = p2_initial
    x[2:18] = alpha.flatten() 
    
    
    return x
    
def x_deconstr(x):
    return x[0], x[1], x[2:18].reshape(4,4)
    

def obj_fun(x, n_clients_per_class):
    p1, p2_initial, alpha = x_deconstr(x) 
    return  - objective_function(p1, p2_initial, alpha, n_clients_per_class)
    
bnds = ((0, None), # p1 prize of first item positive
        (0, None), # p2 should be over d2 but the variable can't be accessed as margin 2 is defined so for now we'll keep it positive
        (0,1), (0,1), (0,1), (0,1), (0,1), (0,1), (0,1), (0,1), (0,1), (0,1), (0,1), (0,1), (0,1), (0,1), (0,1), (0,1), #surement la pire ligne que j'ai écrite de ma vie m'enfin
        
        )



Constraint_Array = np.array([
                            [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #contrainte sur la bonne définition des alpha_1_j
                            [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #contrainte sur le max de promo offrable à la classe 1
                            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], #idem classe 2
                            [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], 
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0], #idem classe 3
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1], #idem classe 4
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                            ])


lb_constraint = np.zeros(8)


c1, c2, c3, c4 = 1, 1, 1, 1 #pour l'instant je les mets à 1 parceque je n'ai pas d'idée et je me demande même si on ne peut pas les passer comme varibales vu
                            #que ça fait en quelque sorte partie des variables à optimiser, mais elles ne seraient pas indépendantes des alpha_j
ub_constraint = np.array([1, c1, 1, c2, 1, c3, 1, c4])

cons1 = so.LinearConstraint(Constraint_Array, lb_constraint, ub_constraint)

x0 = x_constr(100, 100,     
              alpha = np.array([[1., 0., 0., 0.],
                                [1., 0., 0., 0.],
                                [1., 0., 0., 0.],
                                [1., 0., 0., 0.]]) 
                                                )
n_clients_per_class = np.array([50, 20, 10, 5])

res = so.minimize(obj_fun, x0 = x0, args = n_clients_per_class, bounds = bnds, constraints = cons1)

gradient_obj_fun = grad(obj_fun)
gradient_obj_fun(x0, n_clients_per_class)


