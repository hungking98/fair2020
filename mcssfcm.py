import numpy as np
import pandas as pd
import copy

#mc - ssfcm argorithm

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
    U = np.zeros(shape= (n,clus_number))
    return U

def create_U_help(U_fcm):
    U_help = copy.deepcopy(U_fcm)
    n = len(U_fcm)
    numberclus = U_fcm.shape[1]
    for i in range(n):
        for j in range(numberclus):
            if(U_help[i,j]!=max(U_fcm[i])):
                U_help[i,j] = 0.0
    return U_help

def create_label(X):
    n = len(X)
    A = A = np.zeros(n,dtype = int)
    return A

def update_C(X,C,U,m,U_help):
    n = len(X)
    clus_number = len(C)
    for j in range(clus_number):
        tuso = 0.0
        mauso = 0.0
        for i in range(n):
            tuso += (abs(U[i,j]-U_help[i,j])**m[i])*X[i]
            mauso += abs(U[i,j]-U_help[i,j])**m[i]
        C[j] = tuso/mauso
    return C


def update_Uik(X,C,clus_number,m,i,k,U_help):
    tuso = (1-sum(U_help[i]))*(1/(np.linalg.norm(X[i]-C[k]))**(2/(m[i]-1)))
    mauso = 0.0
    for j in range(clus_number):
        mauso+=1/(np.linalg.norm(X[i]-C[j]))**(2/(m[i]-1))
    uik = U_help[i,k] + tuso/mauso
    return uik

def update_U(X,C,U,clus_number,m,U_help):
    n = len(X)
    for i in range(n):
        for k in range(clus_number):
            U[i,k] = update_Uik(X,C,clus_number,m,i,k,U_help)
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

def MC_SSFCM(X,clus_number,m1,m2,alpha,epsilon,U_fcm):
    C = create_Center(X,clus_number)
    U_help = create_U_help(U_fcm)
    U = create_U(X,clus_number)
    label = create_label(X)
    m = mi_fordata(X,m1,m2,alpha,clus_number)
    count = 1
    while True:
        U = update_U(X,C,U,clus_number,m,U_help)
        C_prev = copy.deepcopy(C)
        C = update_C(X,C,U,m,U_help)
        if(check_end_loop(C,C_prev,epsilon)):
            break
        else:
            count +=1
    label = select_Cluster(X,U,label,clus_number)

    return C,label,U
