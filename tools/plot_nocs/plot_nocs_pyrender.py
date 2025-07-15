import numpy as np
import pyrender
import trimesh

def plot_nocs_cube_around_object(mesh_path, scene):
    try:
        # Load the mesh
        mesh = trimesh.load(mesh_path)

        # Add the NOCS cube LineSets to the scene
        for lineset in nocs_cube_linesets:
            scene.add(lineset)

    except Exception as e:
        print("Error loading mesh:", e)

# Create a pyrender scene
scene = pyrender.Scene()

# Plot the mesh using pyrender
mesh_path = "obj_000001.ply"
mesh = trimesh.load(mesh_path)
mesh_node = pyrender.Mesh.from_trimesh(mesh)
scene.add(mesh_node)

# Plot the NOCS cube around the object
plot_nocs_cube_around_object(mesh_path, scene)

# Create a pyrender viewer
viewer = pyrender.Viewer(scene, use_raymond_lighting=True)

# Keep the viewer open
while viewer.is_active:
    viewer.render()
