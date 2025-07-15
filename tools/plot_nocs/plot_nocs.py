import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh
import pyrender

def interpolate_color(start_color, end_color, t):
    """
    Linearly interpolates between two colors.
    """
    return [start_color[i] + t * (end_color[i] - start_color[i]) for i in range(3)]

def plot_cube(ax):
    # Define vertices of the cube
    vertices = np.array([[0, 0, 0],
                         [1, 0, 0],
                         [1, 1, 0],
                         [0, 1, 0],
                         [0, 0, 1],
                         [1, 0, 1],
                         [1, 1, 1],
                         [0, 1, 1]])

    # Define edges of the cube
    edges = [[0, 1], [1, 2], [2, 3], [3, 0],
             [4, 5], [5, 6], [6, 7], [7, 4],
             [0, 4], [1, 5], [2, 6], [3, 7]]

    # Define colors for each vertex based on NOCS principle
    vertex_colors = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
                     (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]

    # Plot edges of the cube with interpolated colors
    for edge in edges:
        start_vertex = vertices[edge[0]]
        end_vertex = vertices[edge[1]]
        start_color = vertex_colors[edge[0]]
        end_color = vertex_colors[edge[1]]
        
        # Generate points along the edge and interpolate colors
        num_points = 10
        edge_points = np.linspace(start_vertex, end_vertex, num_points)
        edge_colors = [interpolate_color(start_color, end_color, i / (num_points - 1)) for i in range(num_points)]
        
        # Plot segments along the edge with interpolated colors
        for i in range(num_points - 1):
            ax.plot3D(*zip(edge_points[i], edge_points[i+1]), color=edge_colors[i])

def plot_mesh(ax, mesh_path):

    # Load the mesh
    mesh = trimesh.load(mesh_path)

    print("Faces array shape:", mesh.faces.shape)
    print("Faces array dtype:", mesh.faces.dtype)

    max_extent = np.max(mesh.vertices, axis=0)
    min_extent = np.min(mesh.vertices, axis=0)
    scale_factor = 1 / np.max(max_extent - min_extent)
    translate_vector = -min_extent * scale_factor

    # Normalize the mesh to fit within a 1x1x1 cube
    mesh.apply_transform(trimesh.transformations.scale_and_translate(scale=scale_factor, translate=translate_vector))

    target_center = np.array([0.5, 0.5, 0.38])
    current_center = np.mean(mesh.vertices, axis=0)
    print(current_center)
    center_translation = target_center - current_center

    # Apply the translation to move the center of the mesh
    mesh.apply_translation(center_translation)

    # Print max and min values of transformed vertices
    print("Max vertex:", np.max(mesh.vertices, axis=0))
    print("Min vertex:", np.min(mesh.vertices, axis=0))

    # Extract vertex colors from the PLY file
    face_colors = mesh.visual.face_colors / 255.0

    colourRGB = np.array((255.0/255.0, 54.0/255.0, 57/255.0, 1.0))

    mpl_mesh = Poly3DCollection(mesh.vertices[mesh.faces], alpha=1.0)
    mpl_mesh.set_facecolor(face_colors)

    try:
        ax.add_collection3d(mpl_mesh)
        
    except Exception as e:
        print("Error loading mesh:", e)

def plot_mesh_with_pyrender(mesh_path):
    try:
        # Load the mesh
        mesh = trimesh.load(mesh_path)

        # Create a pyrender scene
        scene = pyrender.Scene()

        # Create a pyrender mesh
        mesh_node = pyrender.Mesh.from_trimesh(mesh)

        # Add the mesh to the scene
        scene.add(mesh_node)

        # Create a pyrender viewer
        viewer = pyrender.Viewer(scene, use_raymond_lighting=True)

        # Keep the viewer open
        while viewer.is_active:
            viewer.render()

    except Exception as e:
        print("Error loading mesh:", e)

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the cube
mesh_path = "obj_000001.ply" 
plot_mesh(ax, mesh_path)
plot_cube(ax)

# Set aspect ratio
ax.set_box_aspect([1,1,1])

# Remove grid
ax.grid(False)

# Remove axis labels
ax.set_axis_off()

# Show the plot
plt.show()

