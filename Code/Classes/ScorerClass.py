import numpy as np
class scorer:
    def __init__(self):
        self._bde = BDE_c()
        self._gce = GCE_c()
        self._voi = VOI_c()
    def BDE(self,y,pred):
        return self._bde.BDE(y.reshape((-1)),pred.reshape((-1)))
    def GCE(self,y,pred):
        return self._gce.GCE(y.reshape((-1)),pred.reshape((-1)))
    def VOI(self,y,pred):
        return self._voi.VOI(y.reshape((-1)),pred.reshape((-1)))
    def acc(self,y,pred):
        return np.sum(y==pred)/y.size
    def sens(self,y,pred):
        tp = np.sum(y.reshape((-1)) & pred.reshape((-1)))
        p = np.sum(y)
        if p==0:
            return 1.0
        return tp/p
    def spec(self,y,pred):
        tn = np.sum((1-y.reshape((-1))) & (1-pred.reshape((-1))))
        n = np.sum((1-y))
        if n==0:
            return 1.0
        return tn/n

class BDE_c:
    #Needs to be 2D
    def get_boundary(self,a):
        xd = np.diff(a,2,axis=0,append=False,prepend=False)
        yd = np.diff(a,2,axis=1,append=False,prepend=False)
        boundary = xd|yd
        coords = np.argwhere(boundary)
        return coords
    
    def BDE_dds(self,a,b):
        a_bound = self.get_boundary(a)
        b_bound = self.get_boundary(b)
        
        a_mat = np.array([a_bound]*len(b_bound))
        b_mat = np.transpose(np.array([b_bound]*len(a_bound)),(1,0,2))
        
        d_mat = np.sqrt(np.sum((a_mat-b_mat)**2,axis=2))
        v1 = np.sum(np.min(d_mat,axis=0))
        v2 = np.sum(np.min(d_mat,axis=1))
        vout = 0.5*(v1+v2)
        return vout
    
    def BDE(self,y,pred):
        bde = self.BDE_dds(y,pred)
        return bde

class GCE_c:
    #This is for binary only
    def CEl(self,y,pred):
        sze = len(y)
        minv = 1/(4096*4096)
        sigma = np.zeros((sze))
        
        tp = y==pred & pred
        fp = y!=pred & pred
        fn = y!=pred & 1-pred
        tn = y==pred & 1-pred
        
        tpv = np.sum(tp)
        fpv = np.sum(fp)
        fnv = np.sum(fn)
        tnv = np.sum(tn)
        
        ts = max(np.sum(y),minv)
        fs = max(np.sum(1-y),minv)
        
        sigma[tp] = fnv/ts
        sigma[fn] = tpv/ts
        sigma[tn] = fpv/fs
        sigma[fp] = tnv/fs
        return sigma
    
    def CEg(self,y,pred):
        sigma = self.CEl(y,pred)
        return np.sum(sigma)
    
    def GCE(self,y,pred):
        sze = len(y)
        c_si = self.CEg(y,pred)
        c_is = self.CEg(pred,y)
        gce = min(c_si,c_is)
        gce = gce/sze
        return gce
    
    def LCE(self,y,pred):
        sze = len(y)
        c_si = self.CEl(y,pred)
        c_is = self.CEl(pred,y)
        lce = np.sum(np.min((c_si,c_is),axis=0))
        lce = lce/sze
        return lce

class VOI_c:
    def voi_entropy(self,v):
        u_v = np.unique(v)
        size_v = len(v)
        sum_v = 0
        for a in u_v:
            c_v = np.sum(v==a)/size_v
            sum_v+=c_v*np.log(c_v)
        sum_v = -sum_v
        return sum_v
    
    def voi_mutual(self,y,pred):
        min_prob = 1/(len(y))
        tot_sum = 0
        y_uv = np.unique(y)
        pred_uv = np.unique(pred)
        for a in y_uv:
            for b in pred_uv:
                joint = np.sum((y==a)&(pred==b))/len(y)
                joint = max([joint,min_prob])
                ps = np.sum(y==a)/len(y)
                pi = np.sum(pred==b)/len(pred)
                cv = joint*np.log(joint/(pi*ps))
                tot_sum+=cv
        return tot_sum
    
    def VOI(self,y,pred):
        e_s = self.voi_entropy(y)
        e_i = self.voi_entropy(pred)
        g_si = self.voi_mutual(y,pred)
        v = e_s+e_i-2*g_si
        return v
    
    def SVOI(self,y,pred):
        voi = self.VOI(y,pred)
        mvoi = 2*np.log(max(len(np.unique(y)),len(np.unique(pred))))
        svoi = voi/mvoi
        return svoi
