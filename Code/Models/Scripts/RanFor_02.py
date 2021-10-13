#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import cv2
import xgboost as xgb

import sys
sys.path.insert(0,"../Classes")
from ScorerClass import scorer


# In[2]:


def depth_scale(X):
    bsum = np.sum(X,axis=1)
    bsum[bsum==0] = 1
    out = X/np.moveaxis([bsum]*X.shape[1],(0,1),(1,0))
    return out


def load_and_select(path,is_d,inds):
    X = np.load(path)
    X2 = X[:,inds]
    if is_d:
        X2 = depth_scale(X2)
    return X2


def get_data(inds,rgb=True,lab=True,depth=True):
    Xlist = []
    if rgb:
        Xlist.append(load_and_select("../../../Data/Sampled_Features/fs03.npy",False,inds[0][0]))
        Xlist.append(load_and_select("../../../Data/Sampled_Features/fs04.npy",False,inds[0][1]))
        Xlist.append(load_and_select("../../../Data/Sampled_Features/fs05.npy",False,inds[0][2]))
    if lab:
        Xlist.append(load_and_select("../../../Data/Sampled_Features/fs08.npy",False,inds[1][0]))
        Xlist.append(load_and_select("../../../Data/Sampled_Features/fs09.npy",False,inds[1][1]))
        Xlist.append(load_and_select("../../../Data/Sampled_Features/fs10.npy",False,inds[1][2]))
    if depth:
        Xlist.append(load_and_select("../../../Data/Sampled_Features/fs00.npy",True,inds[2][0]))#d
        Xlist.append(load_and_select("../../../Data/Sampled_Features/fs02.npy",True,inds[2][1]))#d
        Xlist.append(load_and_select("../../../Data/Sampled_Features/fs06.npy",True,inds[2][2]))#d
        Xlist.append(load_and_select("../../../Data/Sampled_Features/fs07.npy",True,inds[2][3]))#d
    
    data = np.hstack(Xlist)
    return data


# In[6]:


from sklearn.model_selection import train_test_split
seed = 2666

y    = np.load("../../../Data/Sampled_Features/y.npy")

tr,t = train_test_split([a for a in range(y.shape[0])],random_state=seed)


# In[7]:


def do_score(pred,y):
    scorer_ = scorer()
    print("Acc:",scorer_.acc(y,pred))
    print("Sens:",scorer_.sens(y,pred))
    print("Spec:",scorer_.spec(y,pred))
    print("VOI:",scorer_.VOI(y,pred))
    print("GCE",scorer_.GCE(y,pred))


# In[12]:


from joblib import dump
from sklearn.ensemble import RandomForestClassifier
def do_rf(X,y,tr,t,name):
    save_name = "all_models/rf_"+name+".joblib"
    
    clf = RandomForestClassifier(n_jobs=12)
    clf.fit(X[tr],y[tr])
    
    dump(clf,save_name)
    
    pred = clf.predict(X[t])
    print(name)
    do_score(pred,y[t])


# In[9]:


from joblib import load
inds = load("important_inds.joblib")

X_rgb = get_data(inds,rgb=True,lab=False,depth=False)
X_lab = get_data(inds,rgb=False,lab=True,depth=False)
X_dep = get_data(inds,rgb=False,lab=False,depth=True)
Xf = np.hstack((X_rgb,X_lab,X_dep))


# In[13]:


do_rf(X_rgb,y,tr,t,"RGB")
do_rf(X_lab,y,tr,t,"LAB")
do_rf(X_dep,y,tr,t,"DEP")
do_rf(Xf,y,tr,t,"Full")


# In[ ]:




