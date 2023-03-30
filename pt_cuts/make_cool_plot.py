import os

import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm

input_dir = 'pt_cut_arrays/'
output_dir = 'pt_cut_plots/'
flag = 'train'

pt_cuts = np.append(np.append(np.arange(0,2,0.25),np.arange(2,5,0.5)), np.arange(5,11,1))

#need to define a 'mappable' to use as my colorbar, using the colour and scale that I want
my_cmap = cm.get_cmap('viridis')
sm = plt.cm.ScalarMappable(cmap=my_cmap, norm = plt.Normalize(vmin = 0, vmax = 10))

x_min = 0
x_max = 150

fig,ax = plt.subplots(figsize = (9,6))

for pt_cut in reversed(pt_cuts):
    in_file_name = (
        f"nb_consituents_above_pt{pt_cut}_{flag}.npy"
    )
    input_file = os.path.join(input_dir, in_file_name)

    nb_constituents = np.load(input_file, "r")
    ax.hist(nb_constituents, color=my_cmap(pt_cut/10), range = [0,150], bins = 150, histtype='step')

ax.set_xlim([x_min, x_max])
ax.set_xlabel('Number of constituents')
ax.set_ylabel('Number of jets')
ax.set_title('Constituents per jet for different pt cuts')
ax.grid(linestyle = '--')

cbar = plt.colorbar(sm)
cbar.ax.set_ylabel('$p_T$ cut')

plt.savefig(os.path.join(output_dir, f"cool_plot_{flag}.pdf"))
plt.close()