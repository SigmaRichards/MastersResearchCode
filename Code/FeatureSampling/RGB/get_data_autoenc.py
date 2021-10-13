import sys
sys.path.insert(0,"../Classes")
from DataClass import DataLoader

import numpy as np
import matplotlib.pyplot as plt
import cv2

def split_image(img,n_splits):
    #Assuming squares
    sz_xy = img.shape[0]
    sz_sm = int(sz_xy/n_splits)
    lay = img.shape[2]
    out = np.zeros((n_splits**2,sz_sm,sz_sm,lay))
    for i in range(n_splits):
        for j in range(n_splits):
            c_x = sz_sm*i
            c_y = sz_sm*j
            out[n_splits*i+j,:,:,:] = img[c_x:c_x+sz_sm,c_y:c_y+sz_sm,:]
    return out

def unsplit_image(imgs):
    num_splits = imgs.shape[0]
    s_splits = int(np.sqrt(num_splits))
    sm_res = imgs.shape[1]
    channels = imgs.shape[3]
    
    big_res = sm_res*s_splits
    
    out = np.zeros((big_res,big_res,channels))
    for i in range(s_splits):
        for j in range(s_splits):
            c_x = sm_res*i
            c_y = sm_res*j
            out[c_x:c_x+sm_res,c_y:c_y+sm_res,:]=imgs[s_splits*i+j,:,:,:]
    return out            

def do_enc(model,imgs):
    shape_in = list(imgs.shape)
    shape_enc = (np.prod(imgs.shape[:-3]),imgs.shape[-3],imgs.shape[-2],imgs.shape[-1])
    imgs_enc = model.predict(np.reshape(imgs,shape_enc))
    shape_out = shape_in[:-3]
    shape_out.extend(imgs_enc.shape[-3:])
    out = np.reshape(imgs_enc,tuple(shape_out))
    return out

def increment_counter(counter,shapes):
    for a in range(len(counter)):
        if counter[a]+1<shapes[a]:
            counter[a] = int(counter[a]+1)
            good = True
            break
        elif a+1<len(counter):
            counter[a] = 0
        else:
            good = False
    if good:
        return counter
    else:
        return False

def single_unsplit(imgs):
    shapes = imgs.shape[:-4]
    counter = np.zeros((len(shapes)),int)
    
    shape_out = list(shapes)
    shape_out.extend([imgs.shape[-3]*2,imgs.shape[-2]*2,imgs.shape[-1]])
    
    if len(shapes)>0:
        one_more = True
    else:
        one_more = False
    
    out = np.zeros(tuple(shape_out))
    while any(shapes-(counter+1)):
        c_obj = imgs
        for a in counter:
            c_obj = c_obj[a]
        out[tuple(counter)] = unsplit_image(c_obj)
        counter = increment_counter(counter,shapes)
    if one_more:
        c_obj = imgs
        for a in counter:
            c_obj = c_obj[a]
        out[tuple(counter)] = unsplit_image(c_obj)
        counter = increment_counter(counter,shapes)
    return out

def do_flatten(imgs):
    s1,s2,s3,s4 = imgs.shape
    out = np.reshape(imgs,(s1*s2*s3,s4))
    return out
    

def full_unsplits_res(imgs,res):
    out = imgs
    while len(out.shape) > 4:
        out = single_unsplit(out)
    out = np.asarray([cv2.resize(a,(res,res)) for a in out])
    out = do_flatten(out)
    return out

def scale_for_local(a):
    return (a-np.min(a))/(np.max(a)-np.min(a))

from tensorflow import keras

def get_enc_models():
    fl_128 = keras.models.load_model("encoders/Flat_32x32x32_encoder.h5")
    fl_64 = keras.models.load_model("encoders/Flat_16x16x32_encoder.h5")
    fl_32 = keras.models.load_model("encoders/Flat_8x8x64_encoder.h5")
    fl_16 = keras.models.load_model("encoders/Flat_4x4x64_encoder.h5")
    d_64 = keras.models.load_model("encoders/Depth_16x16x32_encoder.h5")
    d_32 = keras.models.load_model("encoders/Depth_8x8x64_encoder.h5")
    d_16 = keras.models.load_model("encoders/Depth_4x4x64_encoder.h5")
    return [fl_128,fl_64,fl_32,fl_16,d_64,d_32,d_16]

from joblib import load
from sklearn.model_selection import train_test_split

def samp_256(X_f,X_d,encs):
    X_f128 = np.asarray([split_image(a,2) for a in X_f])
    
    X_f64  = np.asarray([[split_image(a,2) for a in b] for b in X_f128])
    X_f32  = np.asarray([[[split_image(a,2) for a in b] for b in c] for c in X_f64])
    X_f16  = np.asarray([[[[split_image(a,2) for a in b] for b in c] for c in d] for d in X_f32])
    
    X_d64 = np.asarray([scale_for_local(a) for a in X_d])
    X_d32 = np.asarray([scale_for_local(split_image(a,2)) for a in X_d64])
    X_d16 = np.asarray([[scale_for_local(split_image(a,2)) for a in b] for b in X_d32])
    
    e_f128= full_unsplits_res(do_enc(encs[0],X_f128),256)
    e_f64 = full_unsplits_res(do_enc(encs[1],X_f64),256)
    e_f32 = full_unsplits_res(do_enc(encs[2],X_f32),256)
    e_f16 = full_unsplits_res(do_enc(encs[3],X_f16),256)
    
    e_d64 = full_unsplits_res(do_enc(encs[4],X_d64),256)
    e_d32 = full_unsplits_res(do_enc(encs[5],X_d32),256)
    e_d16 = full_unsplits_res(do_enc(encs[6],X_d16),256)
    
    stack = np.reshape(np.hstack((e_f128,e_f64,e_f32,e_f16,e_d64,e_d32,e_d16)),(256,256,352))
    
    return stack

def samp_img(flat,depth,encs,y,samp_seed,samp_size):
    #Expecting 1 instance of each, of full size
    #clf = load("AutoEncoder_Depth+Flat_logit_clf.joblib")
    
    stacked = np.zeros((4096,4096,352),np.float32)
    
    flats = split_image(flat,16)
    d_tmp = np.moveaxis([depth],(0,1,2),(2,0,1))
    depths = split_image(d_tmp, 16)
    
    ys = split_image(y,16)
    
    for a in range(256):
        i,j=a//16,a%16
        ir,jr=i*256,j*256
        stacked[ir:ir+256,jr:jr+256,:] = samp_256(flats[a:a+1],depths[a:a+1],encs)
    
    samp = train_test_split([a for a in range(4096*4096)],train_size = samp_size,random_state=samp_seed)[0]
    i,j = [a//4096 for a in samp],[a%4096 for a in samp]
    out = stacked[i,j]
    y_s = y[i,j,0]
    return out,y_s

import tensorflow as tf

dataloader = DataLoader()
dataloader.load_std_folder("../../../Data/STD_Folder")
dataloader.parse_biclass()



samp_seed = 222
samp_size = 15000
x_name  = "X_features_autoencoder.npy"
y_name  = "y.npy"


all_seeds = train_test_split([a for a in range(100000)],train_size = 79,random_state=samp_seed)[0]

from tqdm import tqdm

samp = np.zeros((samp_size*79,352),np.float32)
y = np.zeros((samp_size*79),bool)
encs = get_enc_models()

for index in tqdm(range(79)):
    f_img = dataloader.get_data("flat",[index])[0]/255
    d_img = dataloader.get_data("depth",[index])[0]
    c_img = dataloader.get_data("class",[index])[0]
    samp[index*samp_size:(index+1)*samp_size],y[index*samp_size:(index+1)*samp_size] = samp_img(f_img,d_img,encs,c_img,all_seeds[index],samp_size)

np.save(x_name, samp)
np.save(y_name,y)
