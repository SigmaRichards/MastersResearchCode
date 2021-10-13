#This script has not been formatted for external use
import copy
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

#Data Path
#path = "../../../Data/DATA_FROM_EARLIER_PHOTOS/Textured_OBJ/"
path = "../../../Data/DanielHess_Dataset2/"
detPath = "../../../Data/detrital_mesh2.obj"
path_out = "out/class/"

from os import listdir
from os.path import isfile, join

allfiles = [f for f in listdir(path) if isfile(join(path, f))]
allObjNames = [f for f in allfiles if (f.find(".obj") != -1)]
base_name = [allObjNames[a][:-4] for a in range(len(allObjNames))]

detMesh = o3d.io.read_triangle_mesh(detPath)
detMesh.paint_uniform_color([0,0,0])



def move_forward(vis):
    i_name = path_out + base_name[index] + "_class.png"
    
    vis.capture_screen_image(i_name,True)
    vis.register_animation_callback(None)
    vis.destroy_window()
    return False

#Suppressing output
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

from tqdm import tqdm

for a in tqdm(range(len(allObjNames))):
    index = a
    mesh = o3d.io.read_triangle_mesh(path + allObjNames[index],False)
    mesh.paint_uniform_color([1,1,1])
    
    backMesh = copy.deepcopy(detMesh)
    
    #theta = -0.3233476477503846 #This is the theta we need... just trust me
    theta = -0.0187 #Value for new dataset
    c_center = mesh.get_center()
    
    mesh.rotate(mesh.get_rotation_matrix_from_xyz((0, 0, theta)),center=c_center)
    backMesh.rotate(mesh.get_rotation_matrix_from_xyz((0, 0, theta)),center=c_center)
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=4096,height=4096,visible=False)#Max depth resolution in 1080, for some reason won't render properly for any higher
    vis.add_geometry(mesh)
    ctr = vis.get_view_control()
    ctr.set_zoom(0.5)
    ctr.change_field_of_view(step=-13)
    
    vis.add_geometry(backMesh,reset_bounding_box=False)
    
    opt = vis.get_render_option()
    opt.background_color = np.asarray([1, 0, 0]) #setting background to red
    
    vis.register_animation_callback(move_forward)
    vis.run()
    del opt
    del ctr
    del vis
