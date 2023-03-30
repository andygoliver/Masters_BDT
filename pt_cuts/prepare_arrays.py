import os
import sys
import h5py

import numpy as np
import matplotlib.pyplot as plt

import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "data_file_dir",
    type=str,
    help="Path to directories with .h5 files containing jet image data.",
)
parser.add_argument(
    "output_dir", type=str, help="Directory to save the processed data in."
)
parser.add_argument(
    "plots_dir", type=str, help="Directory to save the plots in."
)
parser.add_argument(
    "--flag",
    type=str,
    default="",
    help="Attach a string to the end of the output file name.",
)

args = parser.parse_args()

pt_cuts = np.append(np.append(np.arange(0,2,0.25),np.arange(2,5,0.5)), np.arange(5,11,1))
classes = np.array(['g', 'q', 'w', 'z', 't'])

def get_file_paths(data_file_dir: str) -> list:
    """Gets path to the data files inside a given directory.

    Args:
        data_file_dir: Path to directory containing the data files.

    Returns:
        Array of paths to the data files themselves.
    """
    file_names = os.listdir(data_file_dir)
    file_paths = [os.path.join(data_file_dir, file_name) for file_name in file_names]

    return file_paths

def get_pts_and_classes(data_path:str):
    """Get the list of pts sorted by constituents in the jets"""
    data = h5py.File(data_path)

    pt_list = data["jetConstituentList"][:, :, 5]
    y_data = data["jets"][:, -6:-1]

    return pt_list, y_data

def get_nb_constituents_after_cut(pt_list: np.ndarray, minimum_pt: float):
    boolean_mask = pt_list[:,:] > minimum_pt
    structure_memory = boolean_mask.sum(axis=1)

    return structure_memory

data_file_paths = get_file_paths(args.data_file_dir)

print('Reading files...\n')

pt_list, y_data = get_pts_and_classes(data_file_paths[0])
for file_path in data_file_paths[1:]:
    add_pt_list, add_y_data= get_pts_and_classes(file_path)
    pt_list = np.concatenate((pt_list, add_pt_list), axis=0)
    y_data = np.concatenate((y_data, add_y_data), axis=0)

n_cut = 0
means = np.array([])
medians = np.array([])

classed_means = np.array([])
classed_medians = np.array([])

for pt_cut in pt_cuts:
    nb_constituents = get_nb_constituents_after_cut(pt_list, pt_cut)
    median_const = np.median(nb_constituents)
    mean_const = np.mean(nb_constituents)

    means = np.append(means, mean_const)
    medians = np.append(medians, median_const)

    # store the means & medians for each class in temporary arrays
    # these will be concatenated into the big ones after looping over the classes
    temp_classed_means = np.array([])
    temp_classed_medians = np.array([])

    out_file_name = (
        f"nb_consituents_above_pt{pt_cut}_{args.flag}"
    )
    output_file = os.path.join(args.output_dir, out_file_name)

    np.save(output_file, nb_constituents)

    # create histogram
    plt.rc("xtick", labelsize=23)
    plt.rc("ytick", labelsize=23)
    plt.rc("axes", titlesize=25)
    plt.rc("axes", labelsize=25)
    plt.rc("legend", fontsize=22)

    plt.figure(figsize=(12, 10))
    plt.hist(
        x=nb_constituents,
        bins=150,
        alpha=1.0,
        histtype="step",
        linewidth=2.5,
        label=f"Median: {median_const}",
        color="#648FFF",
        range = [0,150]
    )
    plt.xlabel("Number of Constituents")
    plt.ylabel("Number of Jets")
    plt.legend()
    plt.gca().set_yscale("log")
    plt.savefig(os.path.join(args.plots_dir, f"constituents_above_pt{pt_cut}_{args.flag}.pdf"))
    plt.close()

    plt.rcdefaults
    plt.figure(figsize=(12, 10))

    # loop over the classes and perform the same actions as before
    for class_index in range(len(classes)):
        classed_nb_constituents = nb_constituents[[np.argmax(y)==class_index for y in y_data]]
        classed_median_const = np.median(classed_nb_constituents)
        classed_mean_const = np.mean(classed_nb_constituents)

        temp_classed_means = np.append(temp_classed_means, classed_mean_const)
        temp_classed_medians = np.append(temp_classed_medians, classed_median_const)
        
        classed_out_file_name = (
            f"nb_{classes[class_index]}_jets_above_pt{pt_cut}_{args.flag}"
        )
        classed_output_file = os.path.join(args.output_dir, classed_out_file_name)
        np.save(classed_output_file,classed_nb_constituents)

        plt.hist(
            x=classed_nb_constituents,
            bins=150,
            alpha=1.0,
            histtype="step",
            linewidth=2.5,
            label=f"{classes[class_index]} jets, median: {classed_median_const}",
            range = [0,150]
        )
    plt.xlabel("Number of Constituents")
    plt.ylabel("Number of Jets")
    plt.legend()
    plt.gca().set_yscale("log")
    plt.savefig(os.path.join(args.plots_dir, f"constituents_above_pt{pt_cut}_{args.flag}_by_class.pdf"))
    plt.close()

    # messy solution to concatenate the means into one big array
    # could of course have 5 predefined arrays, but I liked that this allowed for an arbitrary number of classes
    if np.ndim(classed_means) == 2:
        classed_means = np.concatenate((classed_means, [temp_classed_means]), axis = 0)
    elif np.ndim(classed_means) == 1:
        classed_means = np.concatenate(([classed_means], [temp_classed_means]), axis =1)

    if np.ndim(classed_medians) == 2:
        classed_medians = np.concatenate((classed_medians, [temp_classed_medians]), axis = 0)
    elif np.ndim(classed_medians) == 1:
        classed_medians = np.concatenate(([classed_medians], [temp_classed_medians]), axis =1)

    # creates a counter that displays the current progress
    n_cut += 1
    sys.stdout.write('\r')
    sys.stdout.write(f"Applied cut [{n_cut}/{len(pt_cuts)}]")
    sys.stdout.flush()

print('\n')

# Plot the means and medians for each pt cut
plt.rcdefaults()

plt.plot(pt_cuts, means)
plt.xlabel("$p_T$ cut")
plt.ylabel("Mean constituents per jet")
plt.grid(color='grey', linestyle='--')
plt.savefig(os.path.join(args.plots_dir, f"mean_constituents_{args.flag}.pdf"))
plt.close()

plt.plot(pt_cuts, medians)
plt.xlabel("$p_T$ cut")
plt.ylabel("Median constituents per jet")
plt.grid(color='grey', linestyle='--')
plt.savefig(os.path.join(args.plots_dir, f"median_constituents_{args.flag}.pdf"))
plt.close()

for class_index in range(len(classes)):
    plt.plot(pt_cuts, classed_means[:,class_index], label = f'{classes[class_index]} jets')
plt.plot(pt_cuts, means, label = 'all jets', color = 'black', linewidth = 2)
plt.xlabel("$p_T$ cut")
plt.ylabel("Mean constituents per jet")
plt.grid(color='grey', linestyle='--')
plt.legend()
plt.savefig(os.path.join(args.plots_dir, f"classed_mean_constituents_{args.flag}.pdf"))
plt.close()

for class_index in range(len(classes)):
    plt.plot(pt_cuts, classed_medians[:,class_index], label = f'{classes[class_index]} jets')
plt.plot(pt_cuts, medians, label = 'all jets', color = 'black', linewidth = 2)
plt.xlabel("$p_T$ cut")
plt.ylabel("Median constituents per jet")
plt.grid(color='grey', linestyle='--')
plt.legend()
plt.savefig(os.path.join(args.plots_dir, f"classed_median_constituents_{args.flag}.pdf"))
plt.close()

print(
        "\033[92m"
        + f"Process complete:\n-Successfully saved processed data to {args.output_dir} \U0001F370\n-Succesfully saved plots to {args.plots_dir} \U0001F370"
        + "\033[0m"
    )

