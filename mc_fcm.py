#mc_fcm
import numpy as np
import pandas as pd
import copy


def di_forXi(X,c,i):
    n = len(X)
    D =np.zeros(n)
    for j in range(n):
        D[j] = np.linalg.norm(X[i]-X[j])
    D = np.sort(D)
    K = int(n/c)
    sum_min_d = 0.0
    for j in range(K):
        sum_min_d += D[j]
    return sum_min_d
def mi_fordata(X, m1,m2,alpha,c):
    n = len(X)
    dmax = 0.0
    dmin = 0.0
    D = np.zeros(n)
    for i in range(n):
        D[i] = di_forXi(X,c,i)
    dmin = min(D)
    dmax = max(D)
    m = np.zeros(n)
    for i in range(n):
        m[i] = m1 + (m2-m1)*((D[i]-dmin)/(dmax - dmin))**alpha
    return m


#mc_fcm algorithm
def create_Center(X,clus_number):
    y = np.shape(X)
    C = np.random.random(size = (clus_number,y[1]))
    return C
def create_U(X,clus_number):
    n = len(X)
    U = np.random.random(size= (n,clus_number))
    for i in range(n):
        sum_of_Urow = 0.0
        sum_of_Urow = sum(U[i])
        U[i] = U[i]/sum_of_Urow
    return U

def create_label(X):
    n = len(X)
    A = A = np.zeros(n,dtype = int)
    return A

def update_C(X,C,U,m):
    n = len(X)
    clus_number = len(C)
    for j in range(clus_number):
        tuso = 0.0
        mauso = 0.0
        for i in range(n):
            tuso += ((U[i, j])**m[i])*X[i]
            mauso += (U[i, j])**m[i]
        C[j] = tuso/mauso
    return C
def update_Uik(X,C,clus_number,m,i,k):
    Uik =0.0
    for  j in range(clus_number):
        #Uik += ((np.linalg.norm(X[i]-C[k])) / np.linalg.norm(X[i]-C[j]))**(2/(m[i]-1))
        Uik += ((np.linalg.norm(X[i]-C[k]))/(np.linalg.norm(X[i]-C[j])))**(2/(m[i]-1))
    Uik = 1/Uik
    return Uik

def update_U(X,C,U,clus_number,m):
    n = len(X)
    for i in range(n):
        for k in range(clus_number):
            U[i,k] = update_Uik(X,C,clus_number,m,i,k)
    return U

def check_end_loop(a,b,epsilon):
    if(np.linalg.norm(a-b) < epsilon):
                return True
    else:
                return False

def select_Cluster(X,U,label,clus_number):
    n = len(X)
    for i in range(n):
        for j in range(clus_number):
            if max(U[i]) == U[i,j]:
                label[i] = j
    return label

def mc_fcm(X,clus_number,m1,m2,alpha,epsilon):
    C = create_Center(X,clus_number)
    U = create_U(X,clus_number)
    label = create_label(X)
    m = mi_fordata(X,m1,m2,alpha,clus_number)
    count = 1
    while True:
        U = update_U(X,C,U,clus_number,m)
        C_prev = copy.deepcopy(C)
        C = update_C(X,C,U,m)
        if(check_end_loop(C,C_prev,epsilon)):
            break
        else:
            count +=1
    label = select_Cluster(X,U,label,clus_number)

    return C,label



       


