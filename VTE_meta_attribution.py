import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Parameters
D = 10  # Number of feature dimensions
P = 14  # Patches along one dimension (P x P grid)

# Create a tensor with meaningful shapes in each dimension
meta_attribution = np.zeros((D, P, P))
centers = np.random.randint(0, P, size=(D, 2))  # Randomly shift the center for each dimension
radii = np.random.uniform(1.0, 7.0, size=D)  # Randomize the radii instead of growing with feature index
for i in range(D):
    center_x, center_y = centers[i]
    radius = radii[i]
    for y in range(P):
        for x in range(P):
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            if dist <= radius:
                meta_attribution[i, y, x] = max(0, 2 - (2 * dist / radius))  # Smooth gradient

# Function to update the 3D plot
def update_3D(frame, voxels, fig, ax):
    ax.clear()
    ax.voxels(voxels[:frame+1], facecolors=plt.cm.plasma(voxels[:frame+1]), edgecolors='black')
    ax.set_title(f"3D Meta-attribution: Slices 0-{frame}")
    ax.set_xlim(0, D)
    ax.set_ylim(0, P)
    ax.set_zlim(0, P)

# Prepare animation
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ani = FuncAnimation(fig, update_3D, frames=D, fargs=(meta_attribution, fig, ax), interval=500)

# Save animation as a gif using a relative path
file_path = "3D_meta_attribution_random_radii.gif"
ani.save(file_path, writer='pillow')
plt.close(fig)

file_path
