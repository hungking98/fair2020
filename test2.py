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
df0 = pd.read_csv('./wine.data', sep =",", header =None, names = ["cluster","1","2","3","4","5","6","7","8","9","10","11","12","13"])
A0 = df0['cluster']
del df0['cluster']

X = df0.to_numpy()
#--------------------
C,label,U_fcm = Fcm(X,3,3,0.001)
print(Davies_Bouldin(X,label))
print(SSWC(X,label))
print(PBM(X,label))

print(C)
print("-------------")
C1,label1,U,count = SSFCM(X,3,3,U_fcm,0.001)
print(Davies_Bouldin(X,label1))
print(SSWC(X,label1))
print(PBM(X,label1))

print(C1)
print("-----------------")
C2,label2,U2 = MC_FCM(X,3,3,9.1,2,0.001)
print(Davies_Bouldin(X,label2))
print(SSWC(X,label2))
print(PBM(X,label2))


print("--------------------")
C3,label3,U3,count3 = SSFCM(X,3,3,U2,0.001)
print(Davies_Bouldin(X,label3))
print(SSWC(X,label3))
print(PBM(X,label3))
