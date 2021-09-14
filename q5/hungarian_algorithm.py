import numpy as np 
from scipy.optimize import linear_sum_assignment

def step1(m):
    for i in range(m.shape[0]):
        m[i,:] = m[i,:] - np.min(m[i,:])

def step2(m):
    for j in range(m.shape[1]):
        m[:,j] = m[:,j] - np.min(m[:,j])



def step3(m):
    dim = m.shape[0]
    zeros = {}

    for i in range(dim):
        for j in range(dim):
            if m[i,j] == 0 :
                zeros[i*dim+j] = 1

    rows = np.array([0,1,2,3], dtype = int)
    columns = np.array([], dtype= int)

    #3 lines selected
    for i in range(dim):
        for j in range(dim):

            for k in range(i+1,dim):
                b = True
                for zero in zeros.keys():
                    if zero//dim != i and zero%dim != j and zero//dim != k:
                        b = False
                        break
                if b:
                    rows = np.array([i,k])
                    columns = np.array([j])
                    break

            for k in range(j+1,dim):
                b = True
                for zero in zeros.keys():
                    if zero//dim != i and zero%dim != j and zero%dim != k:
                        b = False
                        break
                if b:
                    rows = np.array([i])
                    columns = np.array([j,k])
                    break

    #2 lines selected
    for i in range(dim):
        for j in range(dim):
            b = True
            for zero in zeros.keys():
                if zero//dim != i and zero%dim != j:
                    b = False
                    break
            if b:
                rows = np.array([i])
                columns = np.array([j])
                break

    return rows, columns


def step5(m, covered_rows, covered_cols): 
    dim0 = m.shape[0]
    dim1 = m.shape[1]
    
    rows = np.linspace(0, dim0-1, dim0).astype(int)
    cols = np.linspace(0, dim1-1, dim1).astype(int)
    
    uncovered_rows = np.setdiff1d(rows, covered_rows).astype(int)
    uncovered_cols = np.setdiff1d(cols, covered_cols).astype(int)
    
    min_val = np.max(m)
    
    for i in uncovered_rows.astype(int):
        for j in uncovered_cols.astype(int):
            if m[i,j] < min_val:
                min_val = m[i,j]
                
    for i in uncovered_rows.astype(int):
        m[i,:] -= min_val
    
    for j in covered_cols.astype(int):
        m[:,j] += min_val
            

def find_rows_single_zero(matrix):
    for i in range(0, matrix.shape[0]):
        if np.sum(matrix[i,:] == 0) == 1:
            j = np.argwhere(matrix[i,:]==0).reshape(-1)
            return(i,j)
    return False
    
def find_cols_single_zero(matrix):
    for j in range(0, matrix.shape[1]):
        if np.sum(matrix[:,j] == 0) == 1:
            i = np.argwhere(matrix[:,j]==0).reshape(-1)
            return(i,j)
    return False    
    
def assignment_single_zero_lines(m, assignment) :
    val = find_rows_single_zero(m)
    while(val):
        i,j = val[0], val[1]
        m[i,j] += 1
        m[:,j] += 1
        assignment[i,j] = 1
        val = find_rows_single_zero(m)
        
    val = find_cols_single_zero(m)  
    while(val):
        i,j = val[0], val[1]
        m[i,:] += 1
        m[i,j] += 1
        assignment[i,j] = 1
        val = find_cols_single_zero(m)
    return assignment


def first_zero(m):
    return np.argwhere(m==0) [0][0], np.argwhere(m==0) [0][1]  


def final_assignment(initial_matrix, m):
    assignment = np.zeros(m.shape, dtype = int)
    assignment = assignment_single_zero_lines(m, assignment)
    while(np.sum(m == 0) > 0):
        i,j = first_zero(m)
        assignment[i,j] = 1
        m[i, :] += 1
        m[:,j] += 1
        assignment = assignment_single_zero_lines(m, assignment)       
    return assignment

def hungarian_algorithm(matrix):
    m = matrix.copy()
    step1(m)
    step2(m)
    n_lines = 0 
    max_length = np.maximum(m.shape[0], m.shape[1])
    alpha = 0
    while n_lines != max_length:
        alpha = alpha + 1
        lines = step3(m)
        
        n_lines = len(lines[0]) + len(lines[1])
        if n_lines != max_length: 
            step5(m, lines[0], lines[1])

    return final_assignment(matrix, m)
    

test = np.array([[ 0, 10, 25, 46],
                 [ 0, 24, 55, 91],
                 [ 0, 23, 49, 77],
                 [ 0, 26, 54, 84]])



