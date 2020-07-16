import numpy as np
import pandas as pd


#Davies -Bouldin Index

def Davies_Bouldin(X,label):
    n = len(X)
    
    c = len(np.unique(label))  #đếm số cụm dựa theo mảng kí tự khác nhau của python
    y = X.shape[1]
    centers = np.zeros(shape = (c,y))
    #một mảng để lưu bán kính trung bình của từng cụm. (bán kính tính từ các điểm đến tâm của cụm đang xét)
    D = np.zeros(c)
    Dj = np.zeros(c)
    #xét từng cụm j 
    for j in range(c):
        Clus_j = np.empty([0,y])
        for i in range(n):
            if(label[i]==j):
                Clus_j = np.append(Clus_j,np.array([X[i]]),axis=0)
        len_Clus_j = len(Clus_j)
        centers[j] = sum(Clus_j)/len_Clus_j
        bankinh =0.0
        for i in range(len_Clus_j):
            bankinh+=np.linalg.norm(Clus_j[i]-centers[j])
        bankinh = bankinh/len_Clus_j
        D[j] = bankinh
    for j in range(c):
        maxDj =0.0 
        for k in range(c):
            if(j!=k):
                temp = (D[j]+D[k])/(np.linalg.norm(centers[j]-centers[k]))
                if(maxDj<temp):
                    maxDj = temp
        Dj[j] = maxDj
    Db =0.0
    Db = sum(Dj)/c
    return Db


#######---------------------------
#SSWC index
def SSWC(X,label):
    n = len(X)
    sothuoctinh = X.shape[1]
    #number cluster
    c = len(np.unique(label))
    a = np.zeros(n)
    b = np.zeros(n)
    s = np.zeros(n)
    for i in range(n):
        b[i] = 100000.0

    for i in range(n):
        count = 0
        for j in range(n):
            if(label[i]==label[j] and (i!=j)):
                a[i]+=np.linalg.norm(X[i]- X[j])
                count+=1
        a[i] = a[i]/count

    for j in range(c):
        Clus_j = np.empty([0,sothuoctinh])
        for i in range(n):
            if(label[i]==j):
               Clus_j = np.append(Clus_j,np.array([X[i]]),axis = 0)
        len_clus = len(Clus_j)
        for i in range(n):
            #xet cac diem va cum khac label:
            temp =0.0
            if(label[i]!=j):
                for k in range(len_clus):
                    temp+=np.linalg.norm(X[i]-Clus_j[k])
                temp = temp/len_clus
                if(temp<b[i]):
                    b[i] = temp
    
    for i in range(n):
        s[i] = (b[i]-a[i])/max(b[i],a[i])
    sswc = sum(s)/n
    return sswc
        



          
###------------------------------
#PBM index
def PBM(X,label):
    n = len(X)
    c = len(np.unique(label))
    y = X.shape[1]
    #mang cac tam cum
    centers = np.zeros(shape = (c,y))
    x_means = sum(X)/n
    E1 = 0.0
    for i in range(n):
        E1+=np.linalg.norm(X[i]-x_means)
    
    Ek = 0.0
    for j in range(c):
        Clus_j = np.empty([0,y])
        for i in range(n):
            if(label[i]==j):
                Clus_j = np.append(Clus_j,np.array([X[i]]),axis=0)
        len_Clus_j = len(Clus_j)
        centers[j] = sum(Clus_j)/len_Clus_j
        #tong khoang cach cho moi diem trong cum den tam cum tuong ung
        temp = 0.0
        for i in range(len_Clus_j):
            temp+=np.linalg.norm(Clus_j[i]-centers[j])
        Ek+=temp
    Dk =0.0
    for i in range(c):
        for j in range(c):
            if(i!=j):
                temp1 = np.linalg.norm(centers[i]-centers[j])
                if(Dk<=temp1):
                    Dk = temp1
    
    Pbm = ((E1*Dk)/(c*Ek))**2

    return Pbm




###------------------
#IFV index
#tham so bao gom du lieu, nhan, do phu thuoc
def IFV(X,U,centers):
    n = len(X)
    c = len(centers)
    SDmax = 0.0
    left = 0.0
    right = 0.0
    for j in range(c):
        temp1 = 0.0
        temp2 = 0.0
        for k in range(n):
            temp1+= (U[k,j])**2
            temp2+= np.log2(U[k,j])
        temp1 = temp1/n
        temp2 = temp2/n 
        temp2 = (np.log2(c)-temp2)**2
        left += temp1*temp2
    left = left/c

    for i in range(c):
        for j in range(c):
            if(i!=j):
                d = (np.linalg.norm(centers[i]-centers[j]))**2
                if(SDmax<=d):
                    SDmax = d
    theta = 0.0
    for j in range(c):
        d = 0.0
        for k in range(n):
            d+=(np.linalg.norm(X[k]-centers[j]))**2
        d = d/n 
        theta+=d
    theta =theta/c
    right = SDmax/theta
    ifv = left*right
    return ifv
            



