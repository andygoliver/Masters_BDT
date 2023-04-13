# Prepare the data for training our machine learning algorithms by selecting certain
# features from the big jet images dataset. Additionally applies some cuts to these
# features. More details about the data set are available at:
# https://github.com/pierinim/tutorials/blob/master/GGI_Jan2021/Lecture1/
# Notebook1_ExploreDataset.ipynb
# The link where one can download this data set is at:
# https://zenodo.org/record/3602260#.YnT0xZpBz0o
import os
import sys
import argparse

import h5py
import numpy as np

import pandas as pd

from terminal_colors import tcols

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
    "--min_pt",
    type=float,
    default=None,
    help="Minimum transverse momentum that the data should have.",
)
parser.add_argument(
    "--max_constituents",
    type=int,
    default=8,
    help="Maximum number of jet constituents data should have.",
)
parser.add_argument(
    "--sorted_feature",
    type=str,
    default="pT",
    help="Feature by which constituents are sorted.",
)
parser.add_argument(
    "--sort_ascending",
    type=bool,
    default=False,
    help="Toggle sort ascending or descending",
)
parser.add_argument(
    "--positive_class",
    type=str,
    default=None,
    help="Positive class for a binary classification.",
)
parser.add_argument(
    "--type",
    type=str,
    default="jedinet",
    choices=["andre", "jedinet"],
    help="The type of feature selection to be employed.",
)
parser.add_argument(
    "--padded_value",
    type=float,
    default=-9999,
    help="The value used to pad the arrays.",
)
parser.add_argument(
    "--include_aggregated_features",
    type=bool,
    default=False,
    help="Choose whether to add aggregated features to the dataset.",
)
parser.add_argument(
    "--flag",
    type=str,
    default="",
    help="Attach a string to the end of the output file name.",
)

classes = np.array([b'g', b'q', b'w', b'z', b't'])

def main(args):
    data_file_paths = get_file_paths(args.data_file_dir)

    print('\nReading files...')

    x_data, y_data = select_features(args.type, data_file_paths[0])
    
    n_file = 1
    # print(f"Read file [{n_file}/{len(data_file_paths)}]")
    
    sys.stdout.write('\r')
    sys.stdout.write(f"Read file [{n_file}/{len(data_file_paths)}]")
    sys.stdout.flush()
    

    for file_path in data_file_paths[1:]:
        n_file +=1

        add_x_data, add_y_data = select_features(args.type, file_path)
        x_data = np.concatenate((x_data, add_x_data), axis=0)
        y_data = np.concatenate((y_data, add_y_data), axis=0)

        sys.stdout.write('\r')
        sys.stdout.write(f"Read file [{n_file}/{len(data_file_paths)}]")
        sys.stdout.flush()
    
    print(tcols.OKGREEN + '\nAll files read' + tcols.ENDC)
    print('=================')

    print(f'Removing zero-padding from input dataset')
    x_data, nb_constituents_precut = remove_zero_padding(x_data)
    print('=================')

    '''
    if args.min_pt is not None:
        pt_idx = get_pt_index(args.type)
        print(f'Cutting transverse momentum for pT > {args.min_pt}...')
        x_data, y_data, nb_constituents_precut = cut_transverse_momentum(x_data, y_data, args.min_pt, pt_idx)

        print('=================')
    '''
    if args.include_aggregated_features:
        print(f'Creating aggregated features...')
        means = aggregate_all_features(x_data, 'mean', args.type)
        sums = aggregate_all_features(x_data, 'sum', args.type)
        # means_ = create_aggregated_feature(x_data, 'mean', 'Delta_R', 16, args.type)
        # sums_ = create_aggregated_feature(x_data, 'sum', 'pT', 16, args.type, sorted_feature='Delta_R', sort_ascending=True)
        print('=================')    

    x_data = sort_data(x_data, args.sorted_feature, args.sort_ascending, args.type)
    print('=================')

    print('Restricting number of constituents...')
    x_data = restrict_nb_constituents(x_data, args.max_constituents, padded_value = args.padded_value)
    # print('-----------------')
    # print(x_data[:3]) 
    # print(type(x_data))
    # print('-----------------')
    print('=================')

    print_data_dimensions(x_data)

    #Create the dataframe by flattening x_data and using appropriate headings
    feature_labels = select_feature_labels(args.type)
    x_heading = generate_X_heading(feature_labels, args.max_constituents)

    x_data = reshape_X(x_data)
    y_data = [classes[np.argmax(i)] for i in y_data]

    df = pd.DataFrame(data = x_data, columns = x_heading)
    
    if args.include_aggregated_features:
        for item in means:
            df[item] = means[item]

        for item in sums:
            df[item] = sums[item]

        df["nb_constituents"] = nb_constituents_precut

        agg_flag = 'agg'

    else:
        agg_flag = 'noagg'
    
    df["class"] = y_data

    if args.positive_class is not None:
        df["class"] = 1*(df["class"] == bytes(args.positive_class, 'utf-8'))

    if args.sort_ascending:
        sort_flag = 'l'
    else:
        sort_flag = 'h'

    '''
    out_file_name = (
        f"jet_images_c{args.max_constituents}_pt{args.min_pt}_{args.type}_sort_{sort_flag}{args.sorted_feature}_pad{args.padded_value}_pc{args.positive_class}_{args.flag}.csv"
    )
    '''

    out_file_name = f"jet_images_c{args.max_constituents}_sort_{sort_flag}{args.sorted_feature}_pc{args.positive_class}_{agg_flag}_{args.flag}.csv"
    output_file = os.path.join(args.output_dir, f"{out_file_name}")

    df.to_csv(output_file, index = False)

    print(
        tcols.OKGREEN
        + f"Successfully saved processed data as {output_file} \U0001F370"
        + tcols.ENDC
    )


def get_pt_index(selection_type: str):
    """Returns the position of the pt in the data array."""
    if selection_type == "jedinet":
        return 5
    if selection_type == "andre":
        return 0


def select_features(choice: str, data_path: str) -> tuple([np.ndarray, np.ndarray]):
    """Choose what feature selection to employ on the data."""
    switcher = {
        "andre": lambda: select_features_andre(data_path),
        "jedinet": lambda: select_features_jedinet(data_path),
    }

    data = switcher.get(choice, lambda: None)()
    if data is None:
        raise TypeError("Feature selection name not valid!")

    return data


def select_features_andre(data_file_path: str) -> tuple([np.ndarray, np.ndarray]):
    """Selects (pT, etarel, phirel) features from an .h5 jet data file and puts
    them into a numpy array. Selects the target (either if the jet comes from a gluon,
    quark, W, Z, or top) corresponding to each event and puts it into a separate
    numpy array.

    Args:
        data_file_path: Path to .h5 file containing jet data.

    Returns:
        Data and target arrays with selected features.
    """
    data = h5py.File(data_file_path)
    x_data = data["jetConstituentList"][:, :, [5, 8, 11]]
    y_data = data["jets"][:, -6:-1]

    return x_data, y_data


def select_features_jedinet(data_file_path) -> tuple([np.ndarray, np.ndarray]):
    """Selects (px, py, pz, E, pT, eta, phi, deltaR, Erel, pTrel, phirel, etarel,
    cos(theta), cos(thetarel), thetarot, phirot) features from an .h5 jet data file and
    puts them into a numpy array. Selects the target (either if the jet comes from a
    gluon, quark, W, Z, or top) corresponding to each event and puts it into a separate
    numpy array.

    Args:
        data_file_path: Path to .h5 file containing jet data.

    Returns:
        Data and target arrays with selected features.
    """
    data = h5py.File(data_file_path)
    x_data = data["jetConstituentList"][:, :, :]
    y_data = data["jets"][:, -6:-1]

    return x_data, y_data

def select_feature_labels(choice: str) -> list[str]:
    """Gets the feature labels for a certain type of selection."""
    jedinet_feature_labels = [
        "px",
        "py",
        "pz",
        "E",
        "E_rel",
        "pT",
        "pT_rel",
        "eta",
        "eta_rel",
        "eta_rot",
        "phi",
        "phi_rel",
        "phi_rot",
        "Delta_R",
        "cos_theta",
        "cos_theta_rel"
    ]
    andre_feature_labels = ["pT", "eta_rel", "phi_rel"]

    switcher = {
        "andre": lambda: andre_feature_labels,
        "jedinet": lambda: jedinet_feature_labels,
    }

    feature_labels = switcher.get(choice, lambda: None)()
    if feature_labels is None:
        raise TypeError("Feature labels name not valid!")

    return feature_labels

def generate_X_heading(feature_labels: list[str], n_constituents: int) -> list[str]:
    '''
    Takes the feature labels (obtained via select_feature_lables) and returns
    a longer list to describe the features of each constituent of the jet.

    This is done since the input array to the bdt contains a single list for 
    each jet (see the reshape_X function).
    '''
    X_heading = ['c0_' + feature_label for feature_label in feature_labels]

    for constituent_index in range(1,n_constituents):
        constituent_feature_labels = [f'c{constituent_index}_' + feature_label for feature_label in feature_labels]
        X_heading = np.concatenate((X_heading, constituent_feature_labels), axis = 0)

    return X_heading

def reshape_X(X_array):
    '''
    Reshape the input array to only have two dimensions
    
    The first axis (different jets) is kept the same, whilst the other columns (here 
    different constituents and their individual features) are combined
    '''
    shape = np.shape(X_array)
    n_events = shape[0]
    total_features = np.prod(shape[1:])
    X_array_reshaped = np.reshape(X_array, (n_events, total_features))
    return X_array_reshaped

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

def remove_zero_padding(
    x_data: np.ndarray,
) -> tuple([list, np.ndarray]):
    """Reject zero-padded constituents.

    Args:
        x_data: Array containing the unprocessed data.

    Returns:
        The processed data array.
    """
    boolean_mask = np.any(x_data, axis = 2)
    structure_memory = boolean_mask.sum(axis=1)
    x_data = np.split(x_data[boolean_mask, :], np.cumsum(structure_memory)[:-1])
    x_data = [jet_const for jet_const in x_data if jet_const.size > 0]

    print(f'Mean constituents per jet after padding removed: {np.mean(structure_memory):.2f}')

    return x_data, structure_memory

def cut_transverse_momentum(
    x_data: np.ndarray,
    y_data: np.ndarray,
    minimum_pt: float,
    pt_index: int,
) -> tuple([list, np.ndarray]):
    """Reject constituents that are below a certain transverse momentum.
    If a jet has no constituents with a momentum above the given threshold, then
    the whole jet is removed.

    Args:
        x_data: Array containing the unprocessed data.
        y_data: Array containing the target.
        minimum_pt: The minimum transverse momentum an event can have (GeV).

    Returns:
        The processed data array together with the processed target.
    """
    boolean_mask = x_data[:, :, pt_index] > minimum_pt
    structure_memory = boolean_mask.sum(axis=1)
    x_data = np.split(x_data[boolean_mask, :], np.cumsum(structure_memory)[:-1])
    x_data = [jet_const for jet_const in x_data if jet_const.size > 0]
    y_data = y_data[structure_memory > 0]

    print(f'Mean constituents per jet after pt cut: {np.mean(structure_memory):.2f}')

    return x_data, y_data, structure_memory

def sort_data(x_data: np.ndarray, sorted_feature: int, ascending: bool, feature_type: str, verbosity:int = 1) -> np.ndarray:
    """Sorts data according to the given feature. This can be highest to lowest (default)
    or lowest to highest.

    Args:
        x_data: Array containing the unsorted data (should have no padding)
        sorted_feature: feature over which data will be sorted
        ascending: boolean to determine sort order (True is lowest to highest)
        feature_type: list of features chosen in this analysis

    Returns:
        The sorted data
    """
    if sorted_feature == "pT" and not ascending:
        sorted_data = x_data
        if verbosity > 0:
            print('Default sorting (by pT) applied.')
    else:
        if verbosity > 0:
            print(f'Sorting data by {sorted_feature} from {ascending*"low to high" + (not ascending)*"high to low"}...')
        try:
            feature_labels  = select_feature_labels(feature_type)
            index = feature_labels.index(sorted_feature)
            
            order = ascending - (not ascending)
            sorted_data = [x_data[i][(order*x_data[i][:, index]).argsort()] for i in range(len(x_data))]
            
            if verbosity > 0:
                print(
                    tcols.OKGREEN 
                    + f'Successfully sorted by {sorted_feature} from {ascending*"low to high" + (not ascending)*"high to low"}.'
                    + tcols.ENDC)

        except:
            sorted_data = x_data

            if verbosity > 0:
                print(
                    tcols.FAIL 
                    + f'Unable to sort by {sorted_feature} from {ascending*"low to high" + (not ascending)*"high to low"}.'
                    + tcols.ENC)
                print(f'Please check that {sorted_feature} is in the {feature_type} list of features.')
                print('Data will be sorted by pT as default.')

    return sorted_data

def restrict_nb_constituents(x_data: np.ndarray, max_constituents: int, padded_value: float = 0.0) -> np.ndarray:
    """Force each jet to have an equal number of constituents. If the jet has more,
    then the ones after the given number are discarded. If the jet has less than the
    number of max constituents, then it is padded with 0 values.

    Args:
        x_data: Data array to be processed.
        max_constituents: Exact number of constituents that a jet should have.
        padded_value: Value used to pad jets

    Returns:
        The data array with a fixed number of constituents per jet.
    """

    nb_constituents = np.array([])

    for jet in range(len(x_data)):
        if x_data[jet].shape[0] >= max_constituents:
            nb_constituents = np.append(nb_constituents, max_constituents)
            x_data[jet] = x_data[jet][:max_constituents, :]
        else:
            nb_constituents = np.append(nb_constituents, x_data[jet].shape[0])
            padding_length = max_constituents - x_data[jet].shape[0]
            x_data[jet] = np.pad(x_data[jet], ((0, padding_length), (0, 0)), constant_values = padded_value)

    print(f'Mean constituents per jet in final dataset: {np.mean(nb_constituents):.2f}')

    return np.array(x_data)

def create_aggregated_feature(x_data, operation, feature, nb_const, feature_type, sorted_feature='pT', sort_ascending=False):
    """Returns a list of aggregated features that can be added to the final dataframe."""
    sorted_data = sort_data(x_data, sorted_feature, sort_ascending, feature_type, verbosity=0)
    
    feature_labels  = select_feature_labels(feature_type)
    try:
        feature_index = feature_labels.index(feature)
    except:
        print(f"'{feature}' not found in list of features!")
        return
    
    switcher = {
        "mean": lambda: [np.mean(jet_const[:nb_const,feature_index]) for jet_const in sorted_data],
        "median": lambda: [np.median(jet_const[:nb_const,feature_index]) for jet_const in sorted_data],
        "min": lambda: [np.amin(jet_const[:nb_const,feature_index]) for jet_const in sorted_data],
        "max": lambda: [np.amax(jet_const[:nb_const,feature_index]) for jet_const in sorted_data],
        "sum": lambda: [np.sum(jet_const[:nb_const,feature_index]) for jet_const in sorted_data],
    }

    aggregated_feature = switcher.get(operation, lambda: None)()
    if aggregated_feature is None:
        raise TypeError(f"'{operation}' not found from list of operations!")
    
    return aggregated_feature

def aggregate_all_features(x_data, operation, feature_type):
    """Returns dictionary with all features aggregated by specific operation."""
    agg_feature_dict = {}
    
    feature_labels  = select_feature_labels(feature_type)

    for feature in feature_labels:
        agg_feature_dict[f'{feature}_{operation}'] = create_aggregated_feature(x_data, operation, feature, None, feature_type)

    return agg_feature_dict

def print_data_dimensions(data: np.ndarray):
    """Prints the dimensions of the data explicitely."""
    print(tcols.OKGREEN + "The processed data has dimensions: " + tcols.ENDC)
    print("--------------")
    print(f"Number of jets = {data.shape[0]}")
    print(f"Number of constituents = {data.shape[1]}")
    print(f"Number of features = {data.shape[2]}")
    print("--------------\n")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
