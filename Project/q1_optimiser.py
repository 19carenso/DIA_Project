# -*- coding: utf-8 -*-
"""
Created on Sat May 15 09:06:20 2021

@author: maxime
"""

import autograd.numpy as np
import scipy.optimize as so
from autograd import grad
from q1_functions import objective_function
    

def obj_fun(p1, p2_initial, alpha, n_clients_per_class): 
    '''
    Input :
        natural variables of the problem
                                
        ---------
    
    Output : 
        Negative value of the objective function corresponding to the negative value of the benefits of the day with the parameters passed.
        
    
    -------------------------------
    
    Objective function used by the solver to find the best parameters. We use the function objective_function of the file q1_no_array.py
    
    '''
    return  - objective_function(p1, p2_initial, alpha, n_clients_per_class) #on compte les sous en millions içi
    

def p1_optimal(p1 = 100, p2 = 100, alpha = [1, 0, 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1, 0., 0., 0.], n_clients_per_class = [50, 20, 10, 5]):
    '''
    Inputs :
        ------------
        
    Output :
        p1 : the best prize found for the first item
        
    '''
    bounds_p1 = so.Bounds(0, np.inf) # p1 prize of first item positive
            
    
    x0 = [p1]
            
    gradient_obj_fun = grad(obj_fun)
    
    res = so.minimize(obj_fun, x0 = x0, args = (p2, alpha, n_clients_per_class), jac = gradient_obj_fun, bounds = bounds_p1)
    
    x_sol = res.x

    return np.round(x_sol[0])