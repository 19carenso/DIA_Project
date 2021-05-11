# -*- coding: utf-8 -*-
"""
Created on Mon May 10 22:39:22 2021

@author: maxime
"""
import numpy as np
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
    Renvoie la marge gagnée si l'item 1 est vendu au prix p2.
    On suppose que chaque item 1 est produit avec un coût constant d1.
    '''
    d2 = 30
    return p2 - d2

def conversion1(p1):
    '''
    Renvoie le taux de conversion pour l'item 1 de chaque classe C_i pour un prix proposé p1.
    On suppose un modèle linéaire.
    '''
    p_max_per_class = np.array([60, 70, 80, 100])
    return np.array([1 - p1/p_max if 1 - p1/p_max >= 0 else 0 for p_max in p_max_per_class])

def conversion2(p2):
    #TODO : assez moche comme définition lol, voir comment rendre ça plus propre
    '''
    Renvoie le taux de conversion pour l'item 2 de chaque classe C_i pour une liste prix proposé p2.
    On suppose un modèle linéaire.

    Parameters
    ----------
    p2 : list
        p2 est typiquement égal à [P0, P1, P2, P3] où les Pi sont les prix réduits
    '''
    p2_initial = p2[0]
    p_max_per_class = np.array([2*p2_initial, 2.5*p2_initial, 4*p2_initial, 5*p2_initial])
    f2 = np.zeros((4,4))
    for j,p in enumerate(p2):
        f2[:,j] = np.array([1 - p/p_max if 1 - p/p_max >= 0 else 0 for p_max in p_max_per_class])
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
        la somme de lignes est donc égalle a 1

    n_clients_per_class : list
        liste indiquant pour chaque classe, le nombre de clients journaliers
    '''
            
    n_clients_1_per_class = np.round(conversion1(p1) * n_clients_per_class) # nombre de clients ayant acheté l'item 1 par classe aujourd'hui

    n_clients_1 = np.sum(np.round(n_clients_1_per_class)) # nombre de clients ayant acheté l'item 1 aujourd'hui 

    P = np.array([0, 0.10, 0.25, 0.40]) # Nos promotions, constantes. 

    promotions = (1-P) * p2_initial # prix proposé en fonction de la réduction. attention c'est donc un vecteur.

    return n_clients_1*margin1(p1) + np.dot(n_clients_1_per_class, np.dot(alpha*conversion2(promotions),margin2(promotions)))


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

test1()