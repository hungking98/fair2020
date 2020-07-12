import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
from fcmeans import FCM
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import silhouette_score
from mc_fcm import *
from index import *

df0 = pd.read_csv('./seeds_dataset.txt', sep ="\t", header =None, names = ["A","B","C","D","E","F","G","cluster"])
A0 = df0['cluster']
del df0['cluster']

X = df0.to_numpy()

C,label = mc_fcm(X,3,1.1,9.1,9.9,0.01)
print("MC_FCM:\n")
print(label)
print(C)

y1 = silhouette_score(X,label)
print(y1)
y2 = davies_bouldin_score(X,label)
print(y2)

z1 = SSWC(X,label)
print(z1)
z2 = Davies_Bouldin(X,label)
print(z2)

z3 = PBM(X,label)
print(z3)