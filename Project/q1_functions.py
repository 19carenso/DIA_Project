# -*- coding: utf-8 -*-
"""
Created on Thu May 13 21:33:08 2021

@author: maxime
"""

import autograd.numpy as np
import matplotlib.pyplot as plt
from numpy.core.function_base import linspace

def margin1(p1):
    '''
    Returns the margin provided by selling item 1 at price p1
    Assuming a constant manufacturing cost
    ''' 
    d1 = 80
    return p1-d1

def margin2(p2):
    '''
    p2 is a list of prices p2 after promotion
    '''
    margin2 = [0]* len(p2)
    d2 = 100
    for i,x in enumerate(p2):
        margin2[i] = p2[i] - d2
    return margin2

def conversion1(p1):
    '''
    Returns the conversion rate for item 1 for each class C_i for price p1    
    '''
    p_max_per_class = [200, 300, 400, 600]
    f1 = [0]*len(p_max_per_class)
    
    for i,p_max in enumerate(p_max_per_class):        
        if 1 - p1/p_max >= 0 :             
            f1[i] = 1 - p1/p_max 
        else:
            f1[i] = 0         
    return f1


def conversion1_mod(p1):
    '''
    To generate a non stationary environment we need to modifiy the conversion rate during some periods
    We do so by changing the maximum price classes are willing to pay for buying item 1.
    '''
    p_max_per_class = [250, 350, 450, 650]
    f1 = [0]*len(p_max_per_class)

    for i,p_max in enumerate(p_max_per_class):        
        if 1 - p1/p_max >= 0 :             
            f1[i] = 1 - p1/p_max 
        else:
            f1[i] = 0         
    return f1

def conversion2(p2):
    '''
    Returns the conversion rates for item 2 for each class C_i for a list of input prices p2
  
    Parameters
    ----------
    p2 : list
        [P0, P1, P2, P3] prices after discount
    
    Output
    ----------
    List of length 16 where the 4th first conversion rates correspond to the conversion rates of class 1, the next fourth correspond to the conversion rates of class 2...
    '''
    
    p_max_per_class = [200, 250, 300, 350]
    f2 = [0] * 16
    
    for j,p_max in enumerate(p_max_per_class):
        f2[4*j:4*(j+1)] = [1 - p/p_max if 1 - p/p_max >= 0 else 0 for p in p2]
    return f2

def objective_function(p1, p2_initial, alpha, n_clients_per_class):
    '''
    Returns the total profit of the day

    Parameters
    ----------
    p1 : int
    
    p2_initial : int
        price of item 2 before discount
    
    alpha : list
        list of length 16 containing the attribution of promo Pj to class C_i
        
    n_clients_per_class : list
        liste indiquant pour chaque classe, le nombre de clients journaliers
    '''
    
    f1 = conversion1(p1)
    
    n_clients_1_per_class = [0] * len(f1) #number of clients per class who bought item 1
    
    for i,f in enumerate(f1):
        n_clients_1_per_class[i] = f1[i] * n_clients_per_class[i] 

    n_clients_1 = np.sum(n_clients_1_per_class) #total number of clients per class who bought item 1

    P = [0, 0.10, 0.20, 0.30] # Promotions

    promotions = [(1-p) * p2_initial for p in P] 
    
    f2 = conversion2(promotions)
    
    benefice2 = 0
    
    for j,m in enumerate(margin2(promotions)): 
        for i in range(4):
            benefice2 += alpha[4*i+j]*f2[4*i+j] * m * n_clients_1_per_class[i]
            
    return n_clients_1 * margin1(p1) + benefice2
    

#Need another objective function for non stationary environment
def objective_function_mod(p1, p2_initial, alpha, n_clients_per_class):
    
    f1 = conversion1_mod(p1)
    
    n_clients_1_per_class = [0] * len(f1)
    
    for i,f in enumerate(f1):
        n_clients_1_per_class[i] = f1[i] * n_clients_per_class[i] 

    n_clients_1 = np.sum(n_clients_1_per_class)  

    P = [0, 0.10, 0.20, 0.30]

    promotions = [(1-p) * p2_initial for p in P]
    
    f2 = conversion2(promotions)
    benefice2 = 0
    
    for j,m in enumerate(margin2(promotions)): 
        for i in range(4):
            benefice2 += alpha[4*i+j]*f2[4*i+j] * m * n_clients_1_per_class[i]
            
    return n_clients_1 * margin1(p1) + benefice2
    






    
def objective_function_cv_rate1_unknown(p1, c1, p2_initial, alpha, n_clients_per_class):
    '''
    Modification of the objective function for q3 when the conversion rates for item 1 is unknown and needs to be estimated
    Returns the profit of the day
    '''

    n_clients_1_per_class = [0] * len(n_clients_per_class)
    
    for i in range(len(n_clients_1_per_class)):
        n_clients_1_per_class[i] = c1 * n_clients_per_class[i]
    n_clients_1 = np.sum(n_clients_1_per_class) 

    
    P = [0, 0.10, 0.20, 0.30] 

    promotions = [(1-p) * p2_initial for p in P]
    
    conv2 = conversion2(promotions) 
    
    benefice2 = 0
    
    for j,m in enumerate(margin2(promotions)):
        for i in range(4): 
            benefice2 += alpha[4*i+j]*conv2[4*i+j] * m * n_clients_per_class[i] 
  
    return margin1(p1) * n_clients_1  + benefice2 
    
    
    
def objective_function_cv_rates_unknown(p1, c1, p2, c2, alpha, n_clients):
    '''
    Modification of the objective function for q4 when the conversion rates for both item 1 & item 2 are unknown and need to be estimated
    Returns the profit of the day
    '''
    profit = 0 
    
    n_clients_1 = c1 * n_clients
    
    profit += margin1(p1) * n_clients_1 
    
    Promos = [0, 0.10, 0.20, 0.30] 
    promotions = [p2 * (1 - P) for P in Promos] 
    for i,a in enumerate(alpha[0:4]):
        
        n_clients_1_i = n_clients_1 * a #
        margin = margin2(promotions)
        profit += margin[i]*c2[i]*n_clients_1_i 
    return profit


def experiment_obj_fun(p1, cv_rate_1, p2_after_promo, cv_rate_2):
    profit = np.sum(cv_rate_1)*margin1(p1)
    margin2_all = margin2(p2_after_promo)
    for i in range(len(cv_rate_2)):
        profit += np.sum(cv_rate_2[i])*margin2_all[i]
    return profit


def profit_per_class(classe,p1, p2_initial,alpha, n_clients_per_class):
    '''
    Needed to define the adjacency matrix for the hungarian algorithm.
    Because we need the profit generated by each class separately.
    '''
    n_clients_item_1  = conversion1(p1)[classe]* n_clients_per_class[classe]
    P = np.array([0, 0.10, 0.20, 0.30])
    promotions = (1-P) * p2_initial
    return n_clients_item_1*margin1(p1) + np.sum(n_clients_item_1*(alpha*np.array(conversion2(promotions)).reshape((4,4))[classe])*margin2(promotions))

