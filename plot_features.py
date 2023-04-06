import os
import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from prepare_data_BDT import get_file_paths, select_features, select_feature_labels
from terminal_colors import tcols

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "data_file_dir",
    type=str,
    help="Path to directories with .h5 files containing jet image data.",
)
parser.add_argument(
    "plots_dir", type=str, help="Directory to save the plots in."
)
args = parser.parse_args()

# define classes for the labels
classes = np.array(['g', 'q', 'w', 'z', 't'])

# Use the method from prepare_data_BDT.py to load in all the data
data_file_paths = get_file_paths(args.data_file_dir)

print('\nReading files...')

x_data, y_data = select_features('jedinet', data_file_paths[0])
n_file = 1

sys.stdout.write('\r')
sys.stdout.write(f"Read file [{n_file}/{len(data_file_paths)}]")
sys.stdout.flush()

for file_path in data_file_paths[1:]:
    n_file +=1

    add_x_data, add_y_data = select_features('jedinet', file_path)
    x_data = np.concatenate((x_data, add_x_data), axis=0)
    y_data = np.concatenate((y_data, add_y_data), axis=0)

    sys.stdout.write('\r')
    sys.stdout.write(f"Read file [{n_file}/{len(data_file_paths)}]")
    sys.stdout.flush()


print(tcols.OKGREEN + '\nAll files read' + tcols.ENDC)
print('=================')

# get list of features
feature_list = select_feature_labels('jedinet')

print("Plotting features")
print("-----------------")

# loop over the features, plotting two histograms in each case
# first histogram is the overall distribution, and the second is split into each class
for feature_index in range(len(feature_list)):
    is_not_padded = np.any(np.array(x_data), axis = 2)

    # plt.hist(x_data[:,:,feature_index].flatten(), histtype='step', density = True, bins=40)
    plt.hist(x_data[is_not_padded][:,feature_index], histtype = 'step', density = True, bins =40)
    plt.xlabel(f"{feature_list[feature_index]}")
    plt.ylabel("Prob density")
    plt.gca().set_yscale("log")
    plt.savefig(os.path.join(args.plots_dir, f"{feature_list[feature_index]}_unpadded.pdf"))
    plt.close()

    for class_index in range(len(classes)):
        boolean_index = [np.argmax(y)==class_index for y in y_data]
        # classed_feature = x_data[:,:,feature_index][boolean_index].flatten()
        classed_feature = x_data[boolean_index][is_not_padded[boolean_index]][:,feature_index]
        plt.hist(classed_feature, histtype='step', density = True, bins=40, label = classes[class_index])

    plt.xlabel(f"{feature_list[feature_index]}")
    plt.ylabel("Prob density")
    plt.gca().set_yscale("log")
    plt.legend()
    plt.savefig(os.path.join(args.plots_dir, f"{feature_list[feature_index]}_by_class_unpadded.pdf"))
    plt.close()  

    print(f"* Created and saved plots of {feature_list[feature_index]}")

print(tcols.OKGREEN + 'All plots created' + tcols.ENDC)
print('=================')