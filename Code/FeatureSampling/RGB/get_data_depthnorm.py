import sys
sys.path.insert(0,"../Classes")
from DataClass import DataLoader

import numpy as np
import cv2
from sklearn.model_selection import train_test_split

def get_norm(gx,gy):
    #getting 0s
    loc_zero = np.argwhere(gy==0)
    loc_non = np.argwhere(gy!=0)
    
    #output array
    out = np.zeros((gx.shape[0],3))
    
    #Dealing with zeros first
    #out[loc_zero,1] = 0 implied
    out[loc_zero,2] = np.sqrt(1/(1+gx[loc_zero]**2))
    out[loc_zero,0] = (-1*gx[loc_zero])*out[loc_zero,2]
    
    #Dealing with non-zeros
    rg = 1/gy[loc_non]
    rg2 = rg**2
    xrg = gx[loc_non]*rg
    xrg2 = xrg**2
    
    out[loc_non,1] = -1*np.sqrt(1/(1+rg2+xrg2))
    out[loc_non,0] = xrg*out[loc_non,1]
    out[loc_non,2] = (-1*out[loc_non,1])/gy[loc_non]
    
    return out

def get_diffs(img):
    x2 = np.diff(img,2)
    x1 = np.diff(img,1)
    xst = x1[:,[0]]
    xen = x1[:,[-1]]
    xs = np.hstack((xst,x2,xen))
    
    
    y2 = np.diff(img,2,axis=0)
    y1 = np.diff(img,1,axis=0)
    yst = y1[[0],:]
    yen = y1[[-1],:]
    ys = np.vstack((yst,y2,yen))
    
    out = np.stack((xs,ys))
    return out

def get_norm_res(img):
    #expect size to be 1024
    xd,yd = get_diffs(img)
    n = np.reshape(get_norm(xd.flatten(),yd.flatten()),(1024,1024,3)).astype(np.float32)
    out = cv2.resize(n,(4096,4096))
    return out

def norm_samp(img,y,samp_seed,samp_size):
    norm_img = get_norm_res(img)
    samp = train_test_split([a for a in range(4096*4096)],train_size = samp_size,random_state=samp_seed)[0]
    i,j = [a//4096 for a in samp],[a%4096 for a in samp]
    out = norm_img[i,j]
    return out,y[i,j,0]

dataloader = DataLoader()
dataloader.load_std_folder("../../../Data/STD_Folder")
dataloader.parse_biclass()





samp_seed = 222
samp_size = 15000
x_name  = "X_depth_norm.npy"
y_name  = "y.npy"





from tqdm import tqdm

all_seeds = train_test_split([a for a in range(100000)],train_size = 79,random_state=samp_seed)[0]

out = np.zeros((samp_size*79,3),np.float32)
ys = np.zeros((samp_size*79),bool)

for a in tqdm(range(79)):
    img = dataloader.get_data("depth",[a])[0]
    yin = dataloader.get_data("class",[a])[0]
    out[a*samp_size:(a+1)*samp_size,:],ys[a*samp_size:(a+1)*samp_size] = norm_samp(img,yin,all_seeds[a],samp_size)

np.save(x_name, out)
np.save(y_name,ys)
