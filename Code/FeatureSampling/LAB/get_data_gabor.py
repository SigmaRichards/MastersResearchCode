import sys
sys.path.insert(0,"../Classes")
from DataClass import DataLoader
from GaborKernelClass import GaborFeatures
from sklearn.model_selection import train_test_split
import numpy as np
import cv2


def con_and_samp(img,y,gf,samp_seed,samp_size):
    con_img = gf.convolve_kernels(img,["red","green","blue"])
    samp = train_test_split([a for a in range(4096*4096)],train_size = samp_size,random_state=samp_seed)[0]
    i,j = [a//4096 for a in samp],[a%4096 for a in samp]
    out = con_img[i,j]
    out = np.reshape(out,(samp_size,out.shape[1]*out.shape[2]))
    return out,y[i,j,0]


dataloader = DataLoader()
dataloader.load_std_folder("../../../Data/STD_Folder")
dataloader.parse_biclass()

samp_seed = 222
samp_size = 15000
x_name  = "X_features_gabor.npy"
y_name  = "y.npy"

all_seeds = train_test_split([a for a in range(100000)],train_size = 79,random_state=samp_seed)[0]



freqs = [1]
theta = 2*np.pi*np.arange(0,8)/8
band  = [0.01,0.1,1]
std   = [1]
off   = 2*np.pi*np.arange(0,4)/4

gf = GaborFeatures()
gf.add_kernels(freqs,theta,band,std,off)

print(len(gf.kernels))


from tqdm import tqdm

out = np.zeros((samp_size*79,len(gf.kernels)*3),np.float32)
ys = np.zeros((samp_size*79),bool)

for a in tqdm(range(79)):
    img = dataloader.get_data("flat",[a])[0]
    img = cv2.cvtColor(img,cv2.COLOR_RGB2Lab)
    yin = dataloader.get_data("class",[a])[0]
    out[a*samp_size:(a+1)*samp_size,:],ys[a*samp_size:(a+1)*samp_size] = con_and_samp(img,yin,gf,all_seeds[a],samp_size)

np.save(x_name, out)
np.save(y_name,ys)
