import os
import sys
import argparse

import numpy as np
import pandas as pd

import tensorflow_decision_forests as tfdf
from tensorflow.keras.models import load_model

from prepare_data_BDT import load_all_files_in_directory, select_feature_labels, get_file_paths, load_data
from prepare_data_BDT import restrict_nb_constituents, reshape_X, generate_X_heading, choose_features, sort_data

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "train_files_dir",
    type=str,
    help="Path to directories with .h5 files containing jet image data for training.",
)
parser.add_argument(
    "test_files_dir",
    type=str,
    help="Path to directories with .h5 files containing jet image data for testing.",
)
parser.add_argument(
    "output_dir", 
    type=str, 
    help="Directory to save the processed data in."
)
parser.add_argument(
    "small_model_dir",
    type=str,
    help="Directory in which to save/find the small model.",
)
parser.add_argument(
    "small_model_name",
    type=str,
    help="Name of small model.",
)
parser.add_argument(
    "positive_class",
    type=str,
    default = 't',
    help="Positive class for a binary classification.",
)
parser.add_argument(
    "--training_split",
    type=float,
    default=0.1,
    help="What fraction of the training data should be left to train the small model.",
)
parser.add_argument(
    "--max_constituents",
    type=int,
    default=16,
    help="Maximum number of jet constituents data should have.",
)
parser.add_argument(
    "--padded_value",
    type=float,
    default=-9999,
    help="The value used to pad the arrays.",
)
parser.add_argument(
    "--aggregation_operations",
    type=str,
    default=None,
    nargs = '+',
    choices = ['mean', 'median', 'max', 'min', 'sum'],
    help="Aggregation operations to perform on the data.",
)
parser.add_argument(
    "--aggregated_feature_number",
    type=int,
    default=None,
    nargs = '+',
    help="Choose how many constituents to aggregate over.",
)

def main(args):
    classes= ['g', 'q', 'w', 'z', 't']
    
    print('\n=================')
    print('Loading and splitting training dataset')
    print('--------------------------------------')
    x_train_full, y_train_full = load_files_in_directory(args.train_files_dir)
    # y_train_full = [np.argmax(i) for i in y_train_full]
    y_train_full = [1*(np.argmax(i) == classes.index(args.positive_class)) for i in y_train_full]

    split_events = int(len(x_train_full)*args.training_split)

    x_train_small = x_train_full[:split_events]
    x_train_large =  x_train_full[split_events:]

    y_train_small = y_train_full[:split_events]
    y_train_large =  y_train_full[split_events:]

    print('=================')
    print('Loading test dataset')
    print('--------------------')
    x_test, y_test= load_files_in_directory(args.test_files_dir)
    y_test = [1*(np.argmax(i) == classes.index(args.positive_class)) for i in y_test]

    print('\n=================\n')

    try:
        small_model = load_model(os.path.join(args.small_model_dir, args.small_model_name))
        print('-----------------')
        print(f"Small model successfully loaded from: {os.path.join(args.small_model_dir, args.small_model_name)}")

    except:
        print(f"Small model not found at: {os.path.join(args.small_model_dir, args.small_model_name)}")
        print("Training new small model")
        print("\n---------------------------------------------------------\n")
        small_model = train_small_model(x_train_small, y_train_small, 
                                        args.small_model_dir, args.small_model_name)
        print("\n---------------------------------------------------------\n")
        
    print('\n=================\n')
    prepare_and_save_dataset(x_train_large, y_train_large, small_model, args.output_dir, "train",
                             include_sparse_attention=True, aggregation_operations= args.aggregation_operations,
                             aggregated_feature_number= args.aggregated_feature_number,
                             max_constituents = args.max_constituents, padded_value= args.padded_value)
    print('\n=================\n')
    prepare_and_save_dataset(x_test, y_test, small_model, args.output_dir, "test",
                             include_sparse_attention=True, aggregation_operations= args.aggregation_operations,
                             aggregated_feature_number= args.aggregated_feature_number,
                             max_constituents = args.max_constituents, padded_value= args.padded_value)

    return

def make_sparse_attention_data(x_data, y_data):
    boolean_mask = np.any(x_data, axis = 2)
    structure_memory = boolean_mask.sum(axis=1)
    x_constituent_data = x_data[boolean_mask, :]

    y_constituent_data = np.array([], dtype=int)
    for i in range(len(structure_memory)):
        y_constituent_data = np.append(y_constituent_data, structure_memory[i]*[y_data[i]])

    return x_constituent_data, y_constituent_data, structure_memory

def make_sparse_attention_x_data(x_data):
    boolean_mask = np.any(x_data, axis = 2)
    structure_memory = boolean_mask.sum(axis=1)
    x_constituent_data = x_data[boolean_mask, :]

    return x_constituent_data, structure_memory

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

def train_small_model(x_train_small, y_train_small, small_model_dir, small_model_name):
    
    x_constituent_small, y_constituent_small, structure_memory_small = make_sparse_attention_data(x_train_small, y_train_small)
    
    train_small_df = pd.DataFrame(x_constituent_small, columns= select_feature_labels("all"))
    train_small_df["class"] = y_constituent_small
    train_small_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_small_df, label="class")

    BDT_hyperparameters = {'num_trees':60,
                           'shrinkage':0.2,
                           'subsample':0.3,
                           'use_hessian_gain':True,
                           'growing_strategy':'BEST_FIRST_GLOBAL',
                           'max_depth':-1,
                           'max_num_nodes':32,
                           'num_threads': 64
                           }
    
    small_model = tfdf.keras.GradientBoostedTreesModel(**BDT_hyperparameters)

    small_model.fit(train_small_ds, verbose = 0)
    small_model.save(small_model_dir + small_model_name)

    return small_model

def add_model_output(x_data_raw, small_model):
    print("Making input for small model")
    x_constituent, structure_memory = make_sparse_attention_x_data(x_data_raw)

    x_constituent_df = pd.DataFrame(x_constituent, columns= select_feature_labels("all"))
    x_constituent_ds = tfdf.keras.pd_dataframe_to_tf_dataset(x_constituent_df)

    y_pred = small_model.predict(x_constituent_ds)

    print("Adding small model predictions to the dataset")
    x_constituent = np.concatenate((x_constituent, np.reshape(y_pred, (len(y_pred),1))), axis = 1)
    x_data = np.split(x_constituent, np.cumsum(structure_memory)[:-1])

    return x_data

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

def prepare_and_save_dataset(x_data, y_data, small_model, output_dir, flag, 
                             feature_selection= "all", include_sparse_attention= True,
                             aggregation_operations: list[int] = None, 
                             aggregated_feature_number: list[int] = None,
                             max_constituents= None, padded_value= -9999.):
    
    print(f'Preparing the {flag} dataset')
    print('------------------------------')
    x_data = add_model_output(x_data, small_model)

    if aggregation_operations is not None:
        print('-----------------')
        print('Creating aggregated features.')
        aggregation_dict = perform_all_aggregations(x_data, aggregation_operations,
                                                    nb_consts = aggregated_feature_number,
                                                    include_sparse_attention=include_sparse_attention)

    print('-----------------')
    print(f'Choosing features according to feature selection: {feature_selection}')
    x_data = choose_features(x_data, feature_selection, 
                             include_sparse_attention= include_sparse_attention)

    print('-----------------')
    print(f'Restricting number of constituents to {max_constituents} in the {flag} dataset')
    x_data = restrict_nb_constituents(x_data, max_constituents, padded_value = padded_value)
    x_data = reshape_X(x_data)

    feature_labels= select_feature_labels(feature_selection, 
                                          include_sparse_attention = include_sparse_attention)
    x_heading = generate_X_heading(feature_labels, max_constituents)

    print('-----------------')
    print(f'Creating pandas dataframe to save {flag} data to csv')

    df = pd.DataFrame(data = x_data, columns = x_heading)
    df["class"]= y_data

    if aggregation_operations is not None:
        print('Adding aggregated features to pandas dataframe')
        df = pd.concat([df, pd.DataFrame(aggregation_dict)], axis =1)
    
        agg_flag = f'agg_{aggregation_operations[0]}'
        for i in range(1,len(aggregation_operations)):
                agg_flag += f'-{aggregation_operations[i]}'
        
        if aggregated_feature_number is not None:
            agg_flag += f'_c{aggregated_feature_number[0]}'
            for i in range(1,len(aggregated_feature_number)):
                agg_flag += f'-{aggregated_feature_number[i]}'
    else:
        agg_flag = 'noagg'

    print('-----------------')

    out_file_name = f"jet_images_c{args.max_constituents}_{feature_selection}_pc{args.positive_class}_SA_{agg_flag}_{flag}.csv"
    output_file = os.path.join(output_dir, out_file_name)
    print(f'Saving {flag} dataset to {output_file}')
    df.to_csv(output_file, index = False)
    return

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)