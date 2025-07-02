import numpy as np
import re
import matplotlib.pyplot as plt

def load_homogeneous_matrices(filepath):
    matrices = []
    with open(filepath, "r") as f:
        text = f.read()

    # find everything between double square brackets
    pattern = r"\[\[(.*?)\]\]"
    blocks = re.findall(pattern, text, re.DOTALL)

    for block in blocks:
        # split lines
        rows = block.strip().split("]\n [")
        mat = []
        for row in rows:
            numbers = [float(x) for x in row.strip().split()]
            mat.append(numbers)
        matrices.append(np.array(mat))
    return matrices
def plot_coordinate_frame(ax, H, label=None, length=0.1):
    """
    Plots a coordinate frame in 3D using a 4x4 homogeneous matrix H.
    """
    origin = H[:3, 3]
    x_axis = H[:3, 0]
    y_axis = H[:3, 1]
    z_axis = H[:3, 2]
    
    ax.quiver(*origin, *x_axis, color='r', length=length, normalize=True)
    ax.quiver(*origin, *y_axis, color='g', length=length, normalize=True)
    ax.quiver(*origin, *z_axis, color='b', length=length, normalize=True)
    
    if label:
        ax.text(*origin, label, fontsize=8)
# Load your matrices
H_C_W_list = load_homogeneous_matrices('./my_examples/H_C_W.txt')
H_C_W = H_C_W_list[0]

H_O_W_list = load_homogeneous_matrices('./my_examples/H_O_W.txt')
H_O_W = H_O_W_list[0:int(len(H_O_W_list)/len(H_C_W_list))]
# print(H_O_W)
# exit()
# num_obj = int(len(H_O_W_list)/len(H_C_W_list))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# plot the camera coordinate frames
plot_coordinate_frame(ax, H_C_W, length=0.1)
# set the view
# plot the object coordinate frames
for idx, H in enumerate(H_O_W):
    plot_coordinate_frame(ax, H, label=f"Obj {idx}", length=0.05)

# set the view
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Camera and Object Coordinate Frames')
ax.set_box_aspect([1,1,1])
plt.show()