#This should load the capsule with the texture in this folder.
#Confirms working version of open3D

import open3d as o3d

mesh = o3d.io.read_triangle_mesh("capsule.obj", True)
mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh])
