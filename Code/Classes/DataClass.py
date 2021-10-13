from os import listdir
from os.path import isfile, isdir, join
from matplotlib import pyplot as plt
import numpy as np
import re

class DataLoader:
    def __init__(self):
        self.base_path = None
        self.flatfiles = None
        self.depthfiles = None
        self.classfiles = None
        self.isParsed = False
        self.isExtended = False
    
    def _load_all_imgs(self, p_list):
        all_imgs = np.asarray([plt.imread(a) for a in p_list])
        return all_imgs
    
    def _is_biclass(self,imgs):
        #np.any seems to be slow, 
        #however imread accounts for 92%
        #of the time of this function
        is_diff = []
        for impath in imgs:
            img = plt.imread(impath)
            is_diff.append(np.any(img != img[0,0,0]))
        return is_diff
    
    def _coord_extend(self,coords):
        ext = coords
        ext = np.append(ext,coords+np.asarray([1,0]),0)
        ext = np.append(ext,coords-np.asarray([1,0]),0)
        ext = np.append(ext,coords+np.asarray([0,1]),0)
        ext = np.append(ext,coords-np.asarray([0,1]),0)
        
        uniq_coord_str = np.unique([str(a[0])+"_"+str(a[1]) for a in ext])
        uniq_coords = [[int(b[0]),int(b[1])] for b in [a.split("_") for a in uniq_coord_str]]
        return uniq_coords
    
    def get_data(self,ftype = "flat",indices = []):
        if ftype == "flat":
            clist = np.asarray(self.flatfiles)
        elif ftype == "class":
            clist = np.asarray(self.classfiles)
        elif ftype == "region":
            if hasattr(self,'regionfiles'):
                clist = np.asarray(self.regionfiles)
            else:
                print("Err: Region not loaded")
                return
        elif ftype == "depth":
            clist = np.asarray(self.depthfiles)
            if len(indices)!=0:
                clist = clist[indices]
            out = np.asarray([np.load(cimg) for cimg in clist])
            return out
        else:
            print("Err: Please set valid type: 'flat','class' or 'depth'")
            return
        
        if len(indices)!=0:
            clist = clist[indices]
        
        out = np.asarray([plt.imread(cimg)*255 for cimg in clist],np.uint8)
        return out
    
    def get_files_from_folder(self, path, ftype=""):
        allfiles = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
        out = [f for f in allfiles if (f.find(ftype) != -1)]
        return out
    
    def order_files_(self):
        filenames = [path.split("/")[-1] for path in self.classfiles]
        all_coords = np.asarray([[int(a) for a in re.findall("\d+",name)] for name in filenames])
        
        flats_ext_path = ["Tile_+"+str(x)+"_+"+str(y)+"_image.png" for x,y in all_coords]
        f_full_path = [join(self.base_path,"flat",path) for path in flats_ext_path]
        self.flatfiles = [a for a in f_full_path if isfile(a)]
        
        depths_ext_path = ["Tile_+"+str(x)+"_+"+str(y)+"_depth.npy" for x,y in all_coords]
        d_full_path = [join(self.base_path,"depth",path) for path in depths_ext_path]
        self.depthfiles = [a for a in d_full_path if isfile(a)]
        
        if hasattr(self,'regionfiles'):
            region_ext_path = ["Tile_+"+str(x)+"_+"+str(y)+"_region.png" for x,y in all_coords]
            r_full_path = [join(self.base_path,"region",path) for path in region_ext_path]
            self.regionfiles = [a for a in r_full_path if isfile(a)]
    
    def load_std_folder(self, path,get_valid_region=False):
        #Expectation that folder has 3 subdirectories
        # - flat, depth, class
        #Looking for data with .png types
        test = isdir(join(path,"flat")) & isdir(join(path,"depth")) & isdir(join(path,"class"))
        
        if not test:
            print("Err: Path given does not follow std structure")
            return
        
        self.flatfiles = self.get_files_from_folder(join(path,"flat"),".png")
        self.depthfiles = self.get_files_from_folder(join(path,"depth"),".npy")
        self.classfiles = self.get_files_from_folder(join(path,"class"),".png")
        
        if get_valid_region:
            self.regionfiles = self.get_files_from_folder(join(path,"region"),".png")
        self.base_path=path
        self.order_files_()
    
    def parse_biclass(self):
        if(self.classfiles == None):
            print("Err: No class paths: cannot check biclass")
            return False
        
        if(self.isParsed):
            print("Err: Classes already parsed")
            return False
        
        is_biclass = self._is_biclass(self.classfiles)
        
        self.isParsed = True
        self.classfiles = np.asarray(self.classfiles)[np.where(is_biclass)]
        
        self.order_files_()
    
    def extend_parsed(self,d_ext = False):
        if not self.isParsed:
            print("Err: Has not been parsed yet")
            return
        if self.isExtended and not d_ext:
            print("Already extended. If you wish to extend again set 'd_ext=True'")
            return
        
        filenames = [path.split("/")[-1] for path in self.classfiles]
        all_coords = np.asarray([[int(a) for a in re.findall("\d+",name)] for name in filenames])
        ext_coords = self._coord_extend(all_coords)
        
        self.isExtended = True
        
        flats_ext_path = ["Tile_+"+str(x)+"_+"+str(y)+"_image.png" for x,y in ext_coords]
        f_full_path = [join(self.base_path,"flat",path) for path in flats_ext_path]
        self.flatfiles = [a for a in f_full_path if isfile(a)]
        
        classes_ext_path = ["Tile_+"+str(x)+"_+"+str(y)+"_class.png" for x,y in ext_coords]
        c_full_path = [join(self.base_path,"class",path) for path in classes_ext_path]
        self.classfiles = [a for a in c_full_path if isfile(a)]
        
        depths_ext_path = ["Tile_+"+str(x)+"_+"+str(y)+"_depth.png" for x,y in ext_coords]
        d_full_path = [join(self.base_path,"depth",path) for path in depths_ext_path]
        self.depthfiles = [a for a in d_full_path if isfile(a)]
