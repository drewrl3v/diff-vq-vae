import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np

bundles = ['PYT_R', 'PYT_L', 'POPT_R', 'POPT_L', 'MCP', 'ILF_R', 'ILF_L', 'IFOF_R', 'IFOF_L', 'FPT_R', 'FPT_L', 'CC_Fr_1']
groups = []
for streams in bundles:
    groups.append(np.load(f'./subject_originals/{streams}.npy'))

recon_groups = []
for streams in bundles:
    recon_groups.append(np.load(f'./subject_recons/{streams}_vq_diff.npy'))

# Font Sizing:
title_font = {
    'fontsize': 16,
    'fontweight': 'bold',
    'color': 'black'
}

# Define labels and corresponding colors (same for both plots)
labels = ['PYT_R', 'PYT_L', 'POPT_R', 'POPT_L', 'MCP', 'ILF_R', 'ILF_L', 'IFOF_R', 'IFOF_L', 'FPT_R', 'FPT_L', 'CC_Fr_1']
colors = ['#1f77b4', '#1f77b4', '#2ca02c', '#2ca02c', '#9467bd', '#c7bb54', '#c7bb54', '#c94754', '#c94754', '#17becf', '#17becf', '#f285e2']

# Ensure there are enough colors for the labels
assert len(colors) >= len(labels), "There are not enough colors for the labels"

# Create a dictionary to map labels to colors
label_color_dict = dict(zip(labels, colors))

# Create a figure with two 3D subplots
fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={'projection': '3d'}, figsize=(12, 6))

# Function to update limits and plot streamlines
def update_limits_and_plot(ax, label, group):
    np_streamlines = group
    lines = [list(zip(fiber[0], fiber[1], fiber[2])) for fiber in np_streamlines]
    color = label_color_dict[label]  # Get color for the current label
    ax.add_collection3d(Line3DCollection(lines, color=color, alpha=0.25, linewidths=0.25))

    # Update the min and max values
    min_x = np.min(np_streamlines[:, 0, :])
    max_x = np.max(np_streamlines[:, 0, :])
    min_y = np.min(np_streamlines[:, 1, :])
    max_y = np.max(np_streamlines[:, 1, :])
    min_z = np.min(np_streamlines[:, 2, :])
    max_z = np.max(np_streamlines[:, 2, :])
    
    return min_x, max_x, min_y, max_y, min_z, max_z

# Plot the first group
limits1 = [np.inf, -np.inf, np.inf, -np.inf, np.inf, -np.inf]
for label, group in zip(labels, groups):
    new_limits = update_limits_and_plot(ax1, label, group)
    limits1 = [min(limits1[i], new_limits[i]) if i % 2 == 0 else max(limits1[i], new_limits[i]) for i in range(6)]

# Set limits for the first subplot
ax1.set_xlim([limits1[0], limits1[1]])
ax1.set_ylim([limits1[2], limits1[3]])
ax1.set_zlim([limits1[4], limits1[5]])

# Plot the second group (recon_groups)
limits2 = [np.inf, -np.inf, np.inf, -np.inf, np.inf, -np.inf]
for label, recon_group in zip(labels, recon_groups):
    new_limits = update_limits_and_plot(ax2, label, recon_group)
    limits2 = [min(limits2[i], new_limits[i]) if i % 2 == 0 else max(limits2[i], new_limits[i]) for i in range(6)]

# Set limits for the second subplot
ax2.set_xlim([limits2[0], limits2[1]])
ax2.set_ylim([limits2[2], limits2[3]])
ax2.set_zlim([limits2[4], limits2[5]])

# Set the view and remove the axes for both subplots
for title, ax in zip(['Original', 'VQ-Diff Recon'],[ax1, ax2]):
    ax.view_init(elev=0, azim=180)
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.zaxis._axinfo["grid"]['linewidth'] = 0.0
    ax.set_title(f'{title}', fontdict=title_font)
    ax.set_axis_off()

plt.tight_layout()
plt.savefig(f"./diagrams/orig_vs_recon.pdf",format='pdf', dpi=800) 
plt.show()