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
from mcssfcm import *
df0 = pd.read_csv('./iris.data', sep =",", header =None, names = ["A","B","C","D","cluster"])
A0 = df0['cluster']
del df0['cluster']

X = df0.to_numpy()
#--------------------
C,label,U_fcm = Fcm(X,3,2,0.001)
C2,label2,U2 = MC_FCM(X,3,1.1,9.1,2,0.001)
C1,label1,U1 = MC_SSFCM(X,3,1.1,9.1,2,0.001,U2)
