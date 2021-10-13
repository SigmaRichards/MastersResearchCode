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


# In[19]:


from time import time
from joblib import dump
def do_xg(X,y,tr,t,name,times):
    dtrain = xgb.DMatrix(X[tr],label = y[tr])
    dtest = xgb.DMatrix(X[t],label=y[t])
    save_name = "all_models/xgb_"+name+".joblib"
    
    start = time()
    
    param = {}
    param['nthread'] = 12
    param['eval_metric'] = 'error'
    param['verbosity']=0
    
    xgb.train(param,dtrain,10)
    time_per_step = (time()-start)/10
    steps = int(times/time_per_step)
    bst = xgb.train(param,dtrain,steps)
    dump(bst,save_name)
    pred = bst.predict(dtest)>0.5
    print(name)
    do_score(pred,y[t])


# In[9]:


from joblib import load
inds = load("important_inds.joblib")

X_rgb = get_data(inds,rgb=True,lab=False,depth=False)
X_lab = get_data(inds,rgb=False,lab=True,depth=False)
X_dep = get_data(inds,rgb=False,lab=False,depth=True)
Xf = np.hstack((X_rgb,X_lab,X_dep))


# In[20]:


do_xg(X_rgb,y,tr,t,"RGB",60*45)
do_xg(X_lab,y,tr,t,"LAB",60*45)
do_xg(X_dep,y,tr,t,"DEP",60*45)
do_xg(Xf,y,tr,t,"Full",60*45)


# In[ ]:




