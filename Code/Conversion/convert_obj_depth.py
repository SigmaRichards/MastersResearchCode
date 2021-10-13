#This script has not been formatted for external use

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

#Data Path
#path = "../../../Data/Tiles/"
#path = "../../../Data/DATA_FROM_EARLIER_PHOTOS/Textured_OBJ/"
path = "../../../Data/DanielHess_Dataset2/"
path_out = "out/"

from os import listdir
from os.path import isfile, join

allfiles = [f for f in listdir(path) if isfile(join(path, f))]
allObjNames = [f for f in allfiles if (f.find(".obj") != -1)]
base_name = [allObjNames[a][:-4] for a in range(len(allObjNames))]

def move_forward(vis):
    d_name = path_out + "depth/" + base_name[index] + "_depth.npy"
    i_name = path_out + "flat/" + base_name[index] + "_image.png"
    
    dep = vis.capture_depth_float_buffer(True)
    np.save(d_name,dep)
    #vis.capture_screen_image(i_name,True)
    vis.register_animation_callback(None)
    vis.destroy_window()
    return False

#Suppressing output
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

from tqdm import tqdm

for a in tqdm(range(len(allObjNames))):
    index = a
    mesh = o3d.io.read_triangle_mesh(path + allObjNames[index],True)
    
    #theta = -0.3233476477503846 #This is the theta we need... just trust me
    theta = -0.0187 #Value for new dataset
    mesh.rotate(mesh.get_rotation_matrix_from_xyz((0, 0, theta)),center=mesh.get_center())
    
    vis = o3d.visualization.Visualizer()
    vis
    vis.create_window(width=1024,height=1024,visible=False)#Max depth resolution is 1080, for some reason won't render properly for any higher
    vis.add_geometry(mesh)
    ctr = vis.get_view_control()
    ctr.change_field_of_view(step=-13)
    ctr.set_zoom(0.5)
    
    vis.register_animation_callback(move_forward)
    vis.run()
    del ctr
    del vis
