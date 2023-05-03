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
    default=16,
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
    default="all",
    help="The type of feature selection to be employed.",
)
parser.add_argument(
    "--padded_value",
    type=float,
    default=-9999,
    help="The value used to pad the arrays.",
)
# parser.add_argument(
#     "--include_aggregated_features",
#     action= "store_true",
#     help="Choose whether to add aggregated features to the dataset.",
# )
parser.add_argument(
    "--aggregation_operations",
    type=str,
    default=None,
    nargs = '+',
    choices = ['mean', 'median', 'max', 'min', 'sum'],
    help="Aggregation operations to perform on the data.",
)
parser.add_argument(
    "--aggregated_feature_selection",
    type=str,
    default=None,
    help="Choose which features to aggreate over.",
)
parser.add_argument(
    "--aggregated_feature_number",
    type=int,
    default=None,
    nargs = '+',
    help="Choose how many constituents to aggregate over.",
)
parser.add_argument(
    "--include_nb_constituents",
    action= "store_true",
    help="Choose whether to add the number of constituents to the dataset as an additional feature.",
)
parser.add_argument(
    "--flag",
    type=str,
    default="",
    help="Attach a string to the end of the output file name.",
)

classes = np.array([b'g', b'q', b'w', b'z', b't'])

def main(args):
    if args.max_constituents == 0 and not args.include_aggregated_features:
        print('----------------------------------------------------------')
        print(
            tcols.FAIL 
            + f'Process failed: you are trying to create an empty dataset!'
            + tcols.ENDC)
        print('----------------------------------------------------------')
        return

    x_data, y_data = load_files_in_directory(args.data_file_dir)
    print('=================')

    x_data, nb_constituents_precut = remove_zero_padding(x_data)
    print('=================')

    # if args.include_aggregated_features:
    #     print(f'Creating aggregated features...')

    #     if args.aggregated_feature_selection == None:
    #         aggregated_feature_selection = args.type

    #     else:
    #         aggregated_feature_selection = args.aggregated_feature_selection

    #     if args.aggregated_feature_number == None:
    #         means = aggregate_all_features(x_data, 'mean', feature_type = aggregated_feature_selection, nb_const = None)
    #         sums = aggregate_all_features(x_data, 'sum', feature_type = aggregated_feature_selection, nb_const = None)

    #     else:
    #         means = {}
    #         sums = {}
    #         for aggregated_feature_number in args.aggregated_feature_number:
    #             add_means = aggregate_all_features(x_data, 'mean', feature_type = aggregated_feature_selection, nb_const = aggregated_feature_number)
    #             add_sums = aggregate_all_features(x_data, 'sum', feature_type = aggregated_feature_selection, nb_const = aggregated_feature_number)

    #             means.update(add_means)
    #             sums.update(add_sums)
        
        # means_ = create_aggregated_feature(x_data, 'mean', 'Delta_R', 16, args.type)
        # sums_ = create_aggregated_feature(x_data, 'sum', 'pT', 16, args.type, sorted_feature='Delta_R', sort_ascending=True)
        # print('=================')   
    
    if args.aggregation_operations is not None:
        print('Creating aggregated features.')
        aggregation_dict = perform_all_aggregations(x_data, args.aggregation_operations,
                                                    nb_consts = args.aggregated_feature_number) 
        print('=================')

    x_data = sort_data(x_data, args.sorted_feature, args.sort_ascending)
    print('=================')
    
    print(f'Choosing features according to feature selection: {args.type}')
    x_data = choose_features(x_data, args.type)
    print('=================')

    if args.max_constituents != 0:
        print('Restricting number of constituents...')
        x_data = restrict_nb_constituents(x_data, args.max_constituents, padded_value = args.padded_value)
        print('=================')

        print_data_dimensions(x_data)

        feature_labels = select_feature_labels(args.type)
        x_heading = generate_X_heading(feature_labels, args.max_constituents)

        x_data = reshape_X(x_data)

        df = pd.DataFrame(data = x_data, columns = x_heading)

    else:
        print('Removing all constituents from the jets.')
        print('=================')

        df = pd.DataFrame()
    
    # if args.include_aggregated_features:
    #     # for item in means:
    #     #     df[item] = means[item]

    #     # for item in sums:
    #     #     df[item] = sums[item]

    #     df = pd.concat([df, pd.DataFrame(means)], axis =1)
    #     df = pd.concat([df, pd.DataFrame(sums)], axis =1)

    #     agg_flag = 'agg'

    #     if args.aggregated_feature_number != None:
    #         agg_flag += f'_c{args.aggregated_feature_number[0]}'
    #         for i in range(1,len(args.aggregated_feature_number)):
    #             agg_flag += f'-{args.aggregated_feature_number[i]}'

    #     if args.aggregated_feature_selection != None:
    #         agg_flag += f'_{args.aggregated_feature_selection}'

    # else:
    #     agg_flag = 'noagg'

    if args.aggregation_operations is not None:
        print('Adding aggregated features to pandas dataframe')
        df = pd.concat([df, pd.DataFrame(aggregation_dict)], axis =1)
    
        agg_flag = f'agg_{args.aggregation_operations[0]}'
        for i in range(1,len(args.aggregation_operations)):
                agg_flag += f'-{args.aggregation_operations[i]}'
        
        if args.aggregated_feature_number is not None:
            agg_flag += f'_c{args.aggregated_feature_number[0]}'
            for i in range(1,len(args.aggregated_feature_number)):
                agg_flag += f'-{args.aggregated_feature_number[i]}'

        if args.aggregated_feature_selection != None:
            agg_flag += f'_{args.aggregated_feature_selection}'
    else:
        agg_flag = 'noagg'
    
    if args.include_nb_constituents:
        df["nb_constituents"] = nb_constituents_precut

    y_data = [classes[np.argmax(i)] for i in y_data]
    df["class"] = y_data

    if args.positive_class is not None:
        df["class"] = 1*(df["class"] == bytes(args.positive_class, 'utf-8'))

    if args.sort_ascending:
        sort_flag = 'l'
    else:
        sort_flag = 'h'

    if args.max_constituents == 0:
        out_file_name = f"jet_images_c{args.max_constituents}_pc{args.positive_class}_{agg_flag}_{args.flag}.csv"
    else:
        out_file_name = f"jet_images_c{args.max_constituents}_{args.type}_{sort_flag}{args.sorted_feature}_pc{args.positive_class}_{agg_flag}_{args.flag}.csv"
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

def load_data(data_path: str) -> tuple([np.ndarray, np.ndarray]):
    """Loads the full dataset."""
    data = h5py.File(data_path)
    x_data = data["jetConstituentList"][:, :, :]
    y_data = data["jets"][:, -6:-1]

    return x_data, y_data

def load_all_files_in_directory(data_file_dir: str, verbosity: int = 1):
    data_file_paths = get_file_paths(data_file_dir)

    if verbosity > 0:
        print('\nReading files...')

    # x_data, y_data = select_features(args.type, data_file_paths[0])
    x_data, y_data = load_data(data_file_paths[0])
    n_file = 1
    
    if verbosity > 0:
        sys.stdout.write('\r')
        sys.stdout.write(f"Read file [{n_file}/{len(data_file_paths)}]")
        sys.stdout.flush()

    for file_path in data_file_paths[1:]:
        n_file +=1

        # add_x_data, add_y_data = select_features(args.type, file_path)
        add_x_data, add_y_data = load_data(file_path)
        x_data = np.concatenate((x_data, add_x_data), axis=0)
        y_data = np.concatenate((y_data, add_y_data), axis=0)

        if verbosity > 0:
            sys.stdout.write('\r')
            sys.stdout.write(f"Read file [{n_file}/{len(data_file_paths)}]")
            sys.stdout.flush()
    
    if verbosity > 0:
        print(tcols.OKGREEN + '\nAll files read' + tcols.ENDC)

    return x_data, y_data

def load_files_in_directory(data_file_dir: str, nb_files: int = None, verbosity: int = 1):
    data_file_paths = get_file_paths(data_file_dir)

    if nb_files == None:
        max_files = len(data_file_paths)
    else:
        max_files = nb_files

    if verbosity > 0:
        print(f'Reading files in {data_file_dir}')

    # x_data, y_data = select_features(args.type, data_file_paths[0])
    x_data, y_data = load_data(data_file_paths[0])
    n_file = 1
    
    if verbosity > 0:
        sys.stdout.write('\r')
        sys.stdout.write(f"Read file [{n_file}/{max_files}]")
        sys.stdout.flush()

    for file_path in data_file_paths[1:nb_files]:
        n_file +=1

        # add_x_data, add_y_data = select_features(args.type, file_path)
        add_x_data, add_y_data = load_data(file_path)
        x_data = np.concatenate((x_data, add_x_data), axis=0)
        y_data = np.concatenate((y_data, add_y_data), axis=0)

        if verbosity > 0:
            sys.stdout.write('\r')
            sys.stdout.write(f"Read file [{n_file}/{max_files}]")
            sys.stdout.flush()
    
    if verbosity > 0:
        print("\033[92m"+
              "\nAll files read"+
              "\033[0m")

    return x_data, y_data

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

def select_feature_labels(choice: str, include_sparse_attention:bool = False) -> list[str]:
    """Gets the feature labels for a certain type of selection."""
    all_feature_labels = [
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
    nopT_feature_labels = [
        "px",
        "py",
        "pz",
        "E",
        "E_rel",
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
    cut1_feature_labels = [
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
    cut2_feature_labels = [
        "E",
        "E_rel",
        "pT",
        "pT_rel",
        "eta_rel",
        "eta_rot",
        "phi",
        "phi_rel",
        "phi_rot",
        "Delta_R",
        "cos_theta_rel"
    ]
    cut3_feature_labels = [
        "E",
        "E_rel",
        "pT_rel",
        "eta_rel",
        "eta_rot",
        "phi",
        "phi_rel",
        "phi_rot",
        "Delta_R",
        "cos_theta_rel"
    ]
    cut4_feature_labels = [
        "E",
        "E_rel",
        "pT",
        "pT_rel",
        "eta_rel",
        "eta_rot",
        "phi_rel",
        "phi_rot",
        "Delta_R",
        "cos_theta_rel"
    ]
    cut5_feature_labels = [
        "E",
        "E_rel",
        "pT",
        "pT_rel",
        "eta_rot",
        "phi_rot",
        "Delta_R",
    ]
    andre_feature_labels = ["pT", "eta_rel", "phi_rel"]

    switcher = {
        "all": lambda: all_feature_labels,
        "andre": lambda: andre_feature_labels,
        "jedinet": lambda: jedinet_feature_labels,
        "nopT": lambda: nopT_feature_labels,
        "cut1": lambda: cut1_feature_labels,
        "cut2": lambda: cut2_feature_labels,
        "cut3": lambda: cut3_feature_labels,
        "cut4": lambda: cut4_feature_labels,
        "cut5": lambda: cut5_feature_labels,
    }

    feature_labels = switcher.get(choice, lambda: None)()
    if feature_labels is None:
        raise TypeError("Feature labels name not valid!")
    
    if include_sparse_attention:
        feature_labels.append("sparse_attention")

    return feature_labels

def get_feature_indices(selection_type: str, include_sparse_attention:bool = False):
    """Returns indices of selected features in data array."""
    indices = []
    full_feature_labels = select_feature_labels('all', include_sparse_attention=include_sparse_attention)

    for feature in select_feature_labels(selection_type, include_sparse_attention=include_sparse_attention):
        index = full_feature_labels.index(feature)
        indices.append(index)

    return indices

def choose_features(x_data, selection_type: str, include_sparse_attention:bool = False):

    if selection_type == 'all':
        selected_data = x_data
    
    else:
        feature_indices = get_feature_indices(selection_type, include_sparse_attention=include_sparse_attention)
        selected_data = [jet[:,feature_indices] for jet in x_data]

    return selected_data

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
    print(f'Removing zero-padding from input dataset...')

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

def sort_data(x_data: np.ndarray, sorted_feature: int, ascending: bool, feature_type: str = 'all', verbosity:int = 1) -> np.ndarray:
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
                    + tcols.ENDC)
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

# def create_aggregated_feature(x_data, operation, feature, nb_const, feature_type = 'all', 
#                               sorted_feature='pT', sort_ascending=False):
#     """Returns a list of aggregated features that can be added to the final dataframe."""
#     sorted_data = sort_data(x_data, sorted_feature, sort_ascending, feature_type, verbosity=0)
    
#     feature_labels  = select_feature_labels(feature_type)
#     try:
#         feature_index = feature_labels.index(feature)
#     except:
#         print(f"'{feature}' not found in list of features!")
#         return
    
#     switcher = {
#         "mean": lambda: [np.mean(jet_const[:nb_const,feature_index]) for jet_const in sorted_data],
#         "median": lambda: [np.median(jet_const[:nb_const,feature_index]) for jet_const in sorted_data],
#         "min": lambda: [np.amin(jet_const[:nb_const,feature_index]) for jet_const in sorted_data],
#         "max": lambda: [np.amax(jet_const[:nb_const,feature_index]) for jet_const in sorted_data],
#         "sum": lambda: [np.sum(jet_const[:nb_const,feature_index]) for jet_const in sorted_data],
#     }

#     aggregated_feature = switcher.get(operation, lambda: None)()
#     if aggregated_feature is None:
#         raise TypeError(f"'{operation}' not found from list of operations!")
    
#     return aggregated_feature

# def aggregate_all_features(x_data, operation: str, feature_type: str = 'all', 
#                            nb_const: int = None, x_data_feature_type: str = 'all'):
#     """Returns dictionary with a selection of features aggregated by specific operation.
#     For more info on feature types see select_feature_labels.
    
#     Args:
#         x_data: data array to be processed
#         operation: aggregation operation used to create the new feature
#         feature_type: selection of features to use for aggregation
#         nb_const: number of features over whcih aggregation is performed
#         x_data_feature_type: selection of features present in x_data
        
#     Returns:
#         dictionary of aggregated features with their corresponding names"""
#     agg_feature_dict = {}
    
#     feature_labels = select_feature_labels(feature_type)

#     if nb_const == None or nb_const == 150:
#         flag = ''
    
#     else:
#         flag = f'_c{nb_const}'

#     for feature in feature_labels:
#         agg_feature_dict[f'{feature}_{operation}{flag}'] = create_aggregated_feature(x_data, operation, feature, nb_const, feature_type = x_data_feature_type)

#     return agg_feature_dict

def create_aggregated_feature(x_data, operation, feature, nb_const, feature_type = 'all', 
                              sorted_feature='pT', sort_ascending=False,
                              include_sparse_attention:bool =False):
    """Returns a list of aggregated features that can be added to the final dataframe."""
    sorted_data = sort_data(x_data, sorted_feature, sort_ascending, feature_type, verbosity=0)
    
    feature_labels  = select_feature_labels(feature_type,
                                            include_sparse_attention= include_sparse_attention)
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

def aggregate_features(x_data, operation: str, feature_type: str = 'all',
                       include_sparse_attention:bool = False,
                       nb_const: int = None, x_data_feature_type: str = 'all'):
    """Returns dictionary with a selection of features aggregated by specific operation.
    For more info on feature types see select_feature_labels.
    
    Args:
        x_data: data array to be processed
        operation: aggregation operation used to create the new feature
        feature_type: selection of features to use for aggregation
        nb_const: number of features over whcih aggregation is performed
        x_data_feature_type: selection of features present in x_data
        
    Returns:
        dictionary of aggregated features with their corresponding names"""
    agg_feature_dict = {}
    
    feature_labels = select_feature_labels(feature_type, 
                                           include_sparse_attention= include_sparse_attention)

    if nb_const == None or nb_const == 150:
        flag = ''
    
    else:
        flag = f'_c{nb_const}'

    for feature in feature_labels:
        agg_feature_dict[f'{feature}_{operation}{flag}'] = create_aggregated_feature(x_data, operation, feature, 
                                                                                     nb_const, 
                                                                                     feature_type = x_data_feature_type,
                                                                                     include_sparse_attention=include_sparse_attention)

    return agg_feature_dict

def perform_all_aggregations(x_data, operations: list[str], feature_type: str = 'all',
                             nb_consts: list[int] = None, x_data_feature_type: str = 'all',
                             include_sparse_attention:bool = False):

    full_aggregation_dict = {}

    for operation in operations:
        if nb_consts == None:
            agg_feature_dict = aggregate_features(x_data, operation, feature_type= feature_type,
                                                  include_sparse_attention= include_sparse_attention,
                                                  x_data_feature_type= x_data_feature_type)
            full_aggregation_dict.update(agg_feature_dict)
        
        else:
            for nb_const in nb_consts:
                agg_feature_dict = aggregate_features(x_data, operation, feature_type= feature_type,
                                                      include_sparse_attention= include_sparse_attention,
                                                    nb_const= nb_const, x_data_feature_type= x_data_feature_type)
                full_aggregation_dict.update(agg_feature_dict)
    
    return full_aggregation_dict

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
