import numpy as np
import pandas as pd
import copy


###FCM argorithm


def create_Center_fcm(X,clus_number):
    y = np.shape(X)
    C = np.random.random(size = (clus_number,y[1]))
    return C
def create_U_fcm(X,clus_number):
    n = len(X)
    U = np.random.random(size= (n,clus_number))
    for i in range(n):
        sum_of_Urow = 0.0
        sum_of_Urow = sum(U[i])
        U[i] = U[i]/sum_of_Urow
    return U

def create_label_fcm(X):
    n = len(X)
    A = A = np.zeros(n,dtype = int)
    return A

def update_C_fcm(X,C,U,m):
    n = len(X)
    clus_number = len(C)
    for j in range(clus_number):
        tuso = 0.0
        mauso = 0.0
        for i in range(n):
            tuso += ((U[i, j])**m)*X[i]
            mauso += (U[i, j])**m
        C[j] = tuso/mauso
    return C

def update_Uik_fcm(X,C,clus_number,m,i,k):
    Uik =0.0
    for  j in range(clus_number):
        
        Uik += ((np.linalg.norm(X[i]-C[k]))/(np.linalg.norm(X[i]-C[j])))**(2/(m-1))
    Uik = 1/Uik
    return Uik

def update_U_fcm(X,C,U,clus_number,m):
    n = len(X)
    for i in range(n):
        for k in range(clus_number):
            U[i,k] = update_Uik_fcm(X,C,clus_number,m,i,k)
    return U

def check_end_loop(a,b,epsilon):
    if(np.linalg.norm(a-b) < epsilon):
                return True
    else:
                return False

def select_Cluster_fcm(X,U,label,clus_number):
    n = len(X)
    for i in range(n):
        for j in range(clus_number):
            if max(U[i]) == U[i,j]:
                label[i] = j
    return label

def Fcm(X,clus_number,m,epsilon):
    C = create_Center_fcm(X,clus_number)
    U = create_U_fcm(X,clus_number)
    label = create_label_fcm(X)
    count = 1
    while True:
        U = update_U_fcm(X,C,U,clus_number,m)
        C_prev = copy.deepcopy(C)
        C = update_C_fcm(X,C,U,m)
        if(check_end_loop(C,C_prev,epsilon)):
            break
        else:
            count +=1
    label = select_Cluster_fcm(X,U,label,clus_number)

    return C,label,U