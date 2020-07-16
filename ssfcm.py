import numpy as np
import pandas as pd
import copy



def create_U_help(U_fcm):
    U_help = copy.deepcopy(U_fcm)
    n = len(U_fcm)
    numberclus = U_fcm.shape[1]
    for i in range(n):
        for j in range(numberclus):
            if(U_help[i,j]!=max(U_fcm[i])):
                U_help[i,j] = 0.0
    return U_help

def creat_U_SSFCM(X,clus_number):
    n = len(X)
    U = np.zeros(shape= (n,clus_number))
    return U
def create_C(X,clus_number):
    n = len(X)
    y = X.shape[1]
    C = np.random.random((clus_number,y))
    return C

def create_label_ssfcm(X):
    n = len(X)
    A = A = np.zeros(n,dtype = int)
    return A

def update_Ukj_SSFCM(X,C,clus_number,m,k,j,U_help):
    tuso = (1-sum(U_help[k]))*((1/(np.linalg.norm(X[k]-C[j])))**(2/(m-1)))
    mauso = 0.0
    for i in range(clus_number):
        mauso+= (1/(np.linalg.norm(X[k]-C[i])))**(2/(m-1))
    ukj = U_help[k,j]+(tuso/mauso)
    return ukj
def update_U_SSFCM(X,C,U,clus_number,m,U_help):
    n = len(X)
    for k in range(n):
        for j in range (clus_number):
            U[k,j] = update_Ukj_SSFCM(X,C,clus_number,m,k,j,U_help)
    return U        

def update_C_SSFCM(X,C,U,m,U_help):
    n = len(X)
    c = len(C)
    
    for j in range(c):
        tuso = 0.0
        mauso =0.0
        for k in range(n):
            tuso+=((abs(U[k,j]-U_help[k,j]))**m)*X[k]
            mauso+=(abs(U[k,j]-U_help[k,j]))**m
            #tuso+=((U[k,j]-U_help[k,j])**m)*X[k]
            #mauso+=(U[k,j]-U_help[k,j])**m

        C[j] = tuso/mauso
    return C

def check_end_loop_ssfcm(a,b,epsilon):

    if((np.linalg.norm(a-b)) < epsilon):
        return True
    else:
        return False


def select_Cluster_ssfcm(X,U,label,clus_number):
    n = len(X)
    for i in range(n):
        for j in range(clus_number):
            if max(U[i]) == U[i,j]:
                label[i] = j
    return label



def SSFCM(X,clus_number,m,U_fcm,epsilon):
    C = create_C(X,clus_number)
    U_help = create_U_help(U_fcm)
    #print(U_help)
    U = creat_U_SSFCM(X,clus_number)
    label = create_label_ssfcm(X)
    count =1
    while True:
        U = update_U_SSFCM(X,C,U,clus_number,m,U_help)
        Cprev = copy.deepcopy(C)
        C = update_C_SSFCM(X,C,U,m,U_help)
        if(check_end_loop_ssfcm(C,Cprev,epsilon)):
            break
        else:
            count +=1
    label = select_Cluster_ssfcm(X,U,label,clus_number)

    return C,label,U,count




        
