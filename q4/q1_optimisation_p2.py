# -*- coding: utf-8 -*-
"""
Created on Sat May 15 09:06:20 2021

@author: maxime
"""

import autograd.numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as so
from autograd import grad, hessian
from q1_no_array import objective_function, margin1, margin2, conversion1, conversion2, test1


def x_constr(p1, alpha):
    '''
    Input :
        p1 : the prize of the first item
        p2 : the prize of the second item
        alpha : the proportion of promotions associated to each class in form of a list of 16 elements. 
        
        --------------
        
    Output : 
        A list to pass to the solver
        
    ------------------------------------------------
    
    Construct a list to pass to the solver with the natural variables of the problem
    
    '''
    x = np.zeros(17)  
    x[0] = p1
    
    x[1:17] = alpha
    
    
    return x
    
def x_deconstr(x):
    '''
    Input : 
        x : A list of 18 variables corresponding to the natural variables of the problem agenced for the solver
        
        ---------
        
    Output :
        p1 : prize of the first item
        p2 : prize of the second item
        alpha : proportions of promotions associated to each class in the form of a list of 16 elements
    
    -------------------------
    
    Return the variable of the solver in the form of the natural variables of the problem
    '''
    return x[0], x[1:17]
    

def obj_fun(x, n_clients_per_class): 
    '''
    Input :
        x : variable of the solver, built by x_constr with the natural variable of the problem
        n_clients_per_class : number of clients per class of the day. 
                                can also be a mean.
                                
        ---------
    
    Output : 
        Negative value of the objective function corresponding to the negative value of the benefits of the day with the parameters passed.
        
    
    -------------------------------
    
    Objective function used by the solver to find the best parameters. We use the function objective_function of the file q1_no_array.py
    
    '''
    p1, alpha = x_deconstr(x) 
    return  - objective_function(p1, alpha, n_clients_per_class) #on compte les sous en millions içi
    

def x_optimal(p1 = 100, alpha = [1, 0, 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1, 0., 0., 0.], n_clients_per_class = [50, 40, 25, 15], p2 = 200):
    '''
    Inputs :
        ------------
        
    Output :
        p1 : the best prize found for the first item
        p2 : the best prize found for the second item
        
        A list of 4 ints corresponding to the best promotions found for the parameters of the problem.
        [3, 2, 0, 0] : corresponding to P3 for class 1, P2 for class 2, P0 for class 3 and class 4
        Depends of the parameters of the problem, like the definition of the conversion rates, the margins and the promotions,
        which can all be found in the q1_no_array file. 
        
    
    '''
    bnds = ((0, None), # p1 prize of first item positive
            
            (0,1), (0,1), (0,1), (0,1), (0,1), (0,1), (0,1), (0,1), (0,1), (0,1), (0,1), (0,1), (0,1), (0,1), (0,1), (0,1), #surement la pire ligne que j'ai écrite de ma vie m'enfin
            )
    
    Constraint_Array_eq = np.array([
                                [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #contrainte sur la bonne définition des alpha_1_j
                                [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], #idem classe 2
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0], #idem classe 3
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1], #idem classe 4
                                ])
        
    Constraint_Array_ineq = np.array([
            [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #contrainte sur le max de promo offrable à la classe 1
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
            ])
    
    
    lb_constraint_eq = np.array([1, 1, 1, 1])
    lb_constraint_ineq = np.array([0, 0, 0, 0])
    
    
    c1, c2, c3, c4 = 1, 1, 1, 1 #pour l'instant je les mets à 1 parceque je n'ai pas d'idée et je me demande même si on ne peut pas les passer comme varibales vu
                                #que ça fait en quelque sorte partie des variables à optimiser, mais elles ne seraient pas indépendantes des alpha_j
                                
    ub_constraint_eq = np.array([1, 1, 1, 1])
    ub_constraint_ineq = np.array([c1, c2, c3, c4])
    
    cons_eq = so.LinearConstraint(Constraint_Array_eq, lb_constraint_eq, ub_constraint_eq)
    cons_ineq = so.LinearConstraint(Constraint_Array_ineq, lb_constraint_ineq, ub_constraint_ineq)
    
    x0 = x_constr(p1, alpha)
        
    gradient_obj_fun = grad(obj_fun)
    #hessian_obj_fun = hessian(obj_fun)
    
    res = so.minimize(obj_fun, x0 = x0, args = n_clients_per_class, jac = gradient_obj_fun, bounds = bnds, constraints = [cons_eq, cons_ineq])
    
    x_sol = res.x

    return np.round(x_sol[0]), [np.argmax(x_sol[2:6]),np.argmax(x_sol[6:10]), np.argmax(x_sol[10:14]), np.argmax(x_sol[14:18])]