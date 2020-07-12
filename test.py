import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
from fcmeans import FCM
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import silhouette_score
from mc_fcm import *
from fcm  import *
from index import *

df0 = pd.read_csv('./iris.data', sep =",", header =None, names = ["A","B","C","D","cluster"])
A0 = df0['cluster']
del df0['cluster']

X = df0.to_numpy()

C,label = mc_fcm(X,3,1.1,9.1,9.9,0.01)
print("MC_FCM:\n")
print(label)
print(C)

C1,label1,count = fcm(X,3,2,0.01)
print(label1)
print(C1)
print(count)

fcm = FCM(n_clusters=3)
fcm.fit(X)
fcm_centers = fcm.centers 
fcm_labels  = fcm.u.argmax(axis=1)
print(fcm_centers)