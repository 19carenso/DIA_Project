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
    Renvoie la marge gagnée si l'item 1 est vendu au prix p1.
    On suppose que chaque item 1 est produit avec un coût constant d1.
    ''' 
    d1 = 20
    return p1-d1

def margin2(p2):
    '''
    Renvoie la marge gagnée si l'item 2 est vendu au prix p2.
    On suppose que chaque item 2 est produit avec un coût constant d1.
    '''
    margin2 = [0]* len(p2)
    d2 = 30
    for i,x in enumerate(p2):
        margin2[i] = p2[i] - d2
    return margin2

def conversion1(p1):
    '''
    Renvoie le taux de conversion pour l'item 1 de chaque classe C_i pour un prix proposé p1.
    On suppose un modèle linéaire.
    '''
    p_max_per_class = [60, 70, 80, 100]
    f1 = [0]*len(p_max_per_class)
    
    for i,p_max in enumerate(p_max_per_class):        
        if 1 - p1/p_max >= 0 :             
            f1[i] = 1 - p1/p_max 
        else:
            f1[i] = 0         
    return f1

def conversion2(p2):
    #TODO : assez moche comme définition lol, voir comment rendre ça plus propre
    '''
    Renvoie le taux de conversion pour l'item 2 de chaque classe C_i pour une liste de prix proposé p2.
    L'output est une liste de taille 16 où les 4 premières entrées correspondent aux taux de conversion 
    de la classe 1, les 4 suivantes de la classe 2, etc..
    
    On suppose un modèle linéaire.

    Parameters
    ----------
    p2 : list
        p2 est typiquement égal à [P0, P1, P2, P3] où les Pi sont les prix réduits
    '''
    
    p_max_per_class = [150, 200, 350, 500] # les utilisateurs sont près à payer bcp plus pour ce bien 
    f2 = [0] * 16
    
    for j,p_max in enumerate(p_max_per_class):
        f2[4*j:4*(j+1)] = [1 - p/p_max if 1 - p/p_max >= 0 else 0 for p in p2]
    return f2

def objective_function(p1, p2_initial, alpha, n_clients_per_class):
    '''
    Renvoie le profit obtenu sur la journée

    Parameters
    ----------
    p1 : int
        prix de l'item 1 (supposé le même pour toutes les classes)
    
    p2_initial : int
        prix de l'item 2 sans réduction (c'est à dire P0)
    
    alpha : list
        tableau dont les cases représentent l'attribution de chaque promotion Pj pour les clients de la classe Ci
        avec les 4 premières cases correspondant à la 1ère classe , les 4 suivantes à la seconde etc.

    n_clients_per_class : list
        liste indiquant pour chaque classe, le nombre de clients journaliers
    '''
    
    
    f1 = conversion1(p1)
    
    n_clients_1_per_class = [0] * len(f1)
    
    for i,f in enumerate(f1):
        n_clients_1_per_class[i] = f1[i] * n_clients_per_class[i] # nombre de clients ayant acheté l'item 1 par classe aujourd'hui

    n_clients_1 = np.sum(n_clients_1_per_class) # nombre de clients ayant acheté l'item 1 aujourd'hui 

    P = [0, 0.10, 0.20, 0.30] # Nos promotions, constantes. 

    promotions = [(1-p) * p2_initial for p in P] # prix proposé en fonction de la réduction. attention c'est donc une liste
    
    f2 = conversion2(promotions) # on s'en sert donc à chaque fois pour calculer les taux de conversion
                                 # autrement on pourrait choisit des constantes arbitraires comme pour conversion1
    
    benefice2 = 0
    
    for j,m in enumerate(margin2(promotions)): #on parcourt sur les promotions et on récupère la marge m sur la jème promo tant qu'on y est 
        for i in range(4): #on parcourt sur les classes
            benefice2 += alpha[4*i+j]*f2[4*i+j] * m * n_clients_1_per_class[i] # je crois que c'est correct, à vérifier.
            
    return n_clients_1 * margin1(p1) + benefice2
    # return n_clients_1*margin1(p1) + np.dot(n_clients_1_per_class, np.dot(alpha*conversion2(promotions),margin2(promotions)))
    
    
def learner_obj_fun_q3(p1, c1, p2_initial, alpha, n_clients_per_class):
    '''
    Renvoie le profit obtenu sur la journée. 
    Le taux de conversion pour l'item 2 n'est pas estimé mais connu de la fonction conversion2
    '''

    n_clients_1_per_class = [0] * len(n_clients_per_class)
    
    for i in range(len(n_clients_1_per_class)):
        n_clients_1_per_class[i] = c1 * n_clients_per_class[i] # nombre de clients ayant acheté l'item 1 par classe aujourd'hui
    n_clients_1 = np.sum(n_clients_1_per_class) # nombre de clients ayant acheté l'item 1 aujourd'hui 

    
    P = [0, 0.10, 0.20, 0.30] # Nos promotions, constantes. 

    promotions = [(1-p) * p2_initial for p in P] # prix proposé en fonction de la réduction. attention c'est donc une liste
    
    conv2 = conversion2(promotions) 
    
    benefice2 = 0
    
    for j,m in enumerate(margin2(promotions)):
        for i in range(4): 
            benefice2 += alpha[4*i+j]*conv2[4*i+j] * m * n_clients_per_class[i] # je crois que c'est correct, à vérifier.
  
    return margin1(p1) * n_clients_1  + benefice2 
    
    
    
def learner_obj_fun(p1, c1, p2, c2, alpha, n_clients):
    '''
    Renvoie le profit obtenue sur la journée en fonction des estimations des taux de conversion c1 et c2 donnée par les lois Beta du Learner, 
    et en fonction du n_clients sur la journée, les clients pas classe n'étant pas connu.
    p1 est liée à c1 car c'est l'un des bras du learner, tandis que p2 lui est constant mais le learner doit toujours approximer c2. 
    petite remarque, p2 n'est donc nécessaire que pour calculer la marge sur l'objet 2.
    La nuance est que c1 est un flottant, tandis que c2 est une liste de 4 flottants car il y a 4 prix pour l'item 2 à approximer vu
    qu'il y a 4 promotions... donc 4 lois beta à update youhou ! 
    Le learner ne dissocie pas les classes donc il propose les promotions indépendamment des classes. 
    Donc par exemple seule la première ligne d'alpha nous suffit. 
    '''
    profit = 0 
    
    n_clients_1 = c1 * n_clients #estimation par le learner du nombre de clients qui vont acheter l'item 1 aujourd'hui
    
    profit += margin1(p1) * n_clients_1 # bénéfice du à la vente de l'item 1.
    
    Promos = [0, 0.10, 0.20, 0.30] # Nos promotions, constantes, similaires à la question précédente
    promotions = [p2 * (1 - P) for P in Promos] # içi le learner ne différencie pas les classe, donc une seule liste suffit
    
    for i,a in enumerate(alpha[0:4]):
        
        n_clients_1_i = n_clients_1 * a # nombre de clients ayant acheté l'item 1 a qui est proposé la ième promo
        margin = margin2(promotions)
        profit += margin[i]*c2[i]*n_clients_1_i #c2[i] étant l'estimation par le learner du taux de conversion de la promo i
    return profit

###########################
##########       ##########
########   TESTS   ########
##########       ##########
###########################

n_clients_per_class = np.zeros(4) #nombre de clients qui viennent dans la journée par classe, initialisé à zéro. Devra être une variable aléatoire par la suite.
means_per_class = np.array([50, 20, 10, 5])
std_per_class = np.array([10, 10, 1, 3])

for i in range(4):
    x = np.random.normal(means_per_class[i], std_per_class[i])
    if x > 0 : n_clients_per_class[i] = np.round(x)


def test1():
    '''
    Trace les fonctions du modèle décrites plus haut
    '''

    #Example
    pmax = 100
    P1 = range(0,pmax)
    Promotions = [50,45,40,35]

    plt.figure(0)
    plt.axes(xlim = (0,pmax), ylim = (0,pmax))
    plt.grid()
    plt.title('Margin on item 1 and 2 depending on the price p')
    plt.xlabel("p")
    plt.ylabel("margin")
    plt.plot(P1, [margin1(p) for p in P1], 'r')
    plt.plot(P1, [margin2(p) for p in P1], 'b')
    plt.legend(["m1", "m2"])

    plt.figure(1)
    plt.axes(xlim = (0,pmax), ylim = (0,1))
    plt.grid()
    plt.title('Proportions of clients of each class buying item 1 at price p')
    plt.xlabel("p")
    plt.ylabel("conversion1")
    plt.plot(P1, [conversion1(p)[0] for p in P1], 'b')
    plt.plot(P1, [conversion1(p)[1] for p in P1], 'g')
    plt.plot(P1, [conversion1(p)[2] for p in P1], 'r')
    plt.plot(P1, [conversion1(p)[3] for p in P1], 'c')
    plt.legend(["Class 1", "Class 2", "Class 3", "Class 4"])

    plt.figure(2)
    plt.axes(xlim = (0,pmax), ylim = (0,1))
    plt.grid()
    plt.title('Proportions of clients of each class buying item 2 at price p')
    plt.xlabel("p")
    plt.ylabel("conversion2")
    plt.plot(Promotions, conversion2(Promotions)[0], 'bo')
    plt.plot(Promotions, conversion2(Promotions)[1], 'go')
    plt.plot(Promotions, conversion2(Promotions)[2], 'ro')
    plt.plot(Promotions, conversion2(Promotions)[3], 'co')
    plt.legend(["Class 1", "Class 2", "Class 3", "Class 4"])

    plt.show()


def test2():
    '''
    Trace la fonction objective (le profit journalier)
    '''
    #TODO : vérifier la fonction objectif, le graphe me parait un peu trop irrégulier...

    pmax = 100
    P1 = range(0,pmax)
    p2_initial = 50
    alpha = [[0.25, 0.25, 0.25, 0.25],
             [0.25, 0.25, 0.25, 0.25],
             [0.25, 0.25, 0.25, 0.25],
             [0.25, 0.25, 0.25, 0.25]]
    n_clients_per_class = [50, 20, 10, 5] #codé en dur pour avoir des expériences répliquables

    plt.figure(0)
    plt.axes(xlim = (0,pmax), ylim = (0, 1000))
    plt.grid()
    plt.title('Total profit depending on the price of item 1')
    plt.xlabel("p")
    plt.ylabel("profit")
    plt.plot(P1, [objective_function(p, p2_initial, alpha, n_clients_per_class) for p in P1], 'r')
    plt.legend(["profit"])
    plt.show()


def test3():
    '''
    Teste une seule valeur de la fonction objectif
    '''

    p1 = 50
    p2_initial = 50
    alpha = [[0.25, 0.25, 0.25, 0.25],
             [0.25, 0.25, 0.25, 0.25],
             [0.25, 0.25, 0.25, 0.25],
             [0.25, 0.25, 0.25, 0.25]]
    n_clients_per_class = [50, 20, 10, 5] #codé en dur pour avoir des expériences répliquables
    res = objective_function(p1, p2_initial, alpha, n_clients_per_class)

    print(res)
