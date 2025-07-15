import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh
from matplotlib.animation import FuncAnimation

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

def plot_mesh(ax, mesh_path, angle):
    # Load the mesh
    mesh = trimesh.load(mesh_path)

    max_extent = np.max(mesh.vertices, axis=0)
    min_extent = np.min(mesh.vertices, axis=0)
    scale_factor = 1 / np.max(max_extent - min_extent)
    translate_vector = -min_extent * scale_factor

    # Normalize the mesh to fit within a 1x1x1 cube
    mesh.apply_transform(trimesh.transformations.scale_and_translate(scale=scale_factor, translate=translate_vector))

    target_center = np.array([0.5, 0.5, 0.38])
    current_center = np.mean(mesh.vertices, axis=0)
    center_translation = target_center - current_center

    # Apply the translation to move the center of the mesh
    mesh.apply_translation(center_translation)

    # Extract face colors from the PLY file
    face_colors = mesh.visual.face_colors / 255.0

    # Plot the mesh faces with individual colors
    mpl_mesh = Poly3DCollection(mesh.vertices[mesh.faces], alpha=1.0)
    mpl_mesh.set_facecolor(face_colors)

    ax.add_collection3d(mpl_mesh)

    # Set aspect ratio
    ax.set_box_aspect([1, 1, 1])

    # Remove grid
    ax.grid(False)

    # Remove axis labels
    ax.set_axis_off()

    ax.view_init(elev=30., azim=angle)  # Set the viewpoint angle with elevation

def update(frame):
    ax.cla()
    angle = 360 * frame / frames  # Cover the full viewing hemisphere
    plot_mesh(ax, mesh_path, angle)
    plot_cube(ax)
    print(f"Rendering frame {frame + 1}/{frames}", end='\r')

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

mesh_path = "obj_000001.ply"
frames = 60  # Number of frames for the animation

# Create the animation
ani = FuncAnimation(fig, update, frames=frames, interval=50)

# Save the animation as a GIF
ani.save('mesh_rotation_cube.gif', writer='pillow', fps=15)
