import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import silhouette_score
from mc_fcm import *
from fcm  import *
from fcmeans import FCM
from index import *
from ssfcm import *
df0 = pd.read_csv('./iris.data', sep =",", header =None, names = ["A","B","C","D","cluster"])
A0 = df0['cluster']
del df0['cluster']

X = df0.to_numpy()
#--------------------
C,label,U_fcm = Fcm(X,3,2,0.001)
print(Davies_Bouldin(X,label))
print(SSWC(X,label))
print(PBM(X,label))
print(IFV(X,U_fcm,C))
print("-------------")
C1,label1,U,count = SSFCM(X,3,2,U_fcm,0.001)
print(Davies_Bouldin(X,label1))
print(SSWC(X,label1))
print(PBM(X,label1))
print(IFV(X,U,C1))
print("-----------------")
C2,label2,U2 = MC_FCM(X,3,3,9.1,2,0.001)
print(Davies_Bouldin(X,label2))
print(SSWC(X,label2))
print(PBM(X,label2))
print(IFV(X,U2,C2))

print("--------------------")
C3,label3,U3,count3 = SSFCM(X,3,2,U2,0.001)
print(Davies_Bouldin(X,label3))
print(SSWC(X,label3))
print(PBM(X,label3))
print(IFV(X,U3,C3))

print("-------------------")
print(C)
print(C1)
print(C2)
print(C3)



