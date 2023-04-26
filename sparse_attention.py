import os
import sys
import argparse

import numpy as np
import pandas as pd

import tensorflow_decision_forests as tfdf

from prepare_data_BDT import load_all_files_in_directory, select_feature_labels, get_file_paths, load_data
from prepare_data_BDT import restrict_nb_constituents, reshape_X, generate_X_heading

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
    "output_dir", type=str, help="Directory to save the processed data in."
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

def main(args):
    classes= ['g', 'q', 'w', 'z', 't']

    x_train_full, y_train_full = load_files_in_directory(args.train_files_dir)
    # y_train_full = [np.argmax(i) for i in y_train_full]
    y_train_full = [1*(np.argmax(i) == classes.index(args.positive_class)) for i in y_train_full]

    split_events = int(len(x_train_full)*args.training_split)

    x_train_small = x_train_full[:split_events]
    x_train_large =  x_train_full[split_events:]

    y_train_small = y_train_full[:split_events]
    y_train_large =  y_train_full[split_events:]

    x_constituent_small, y_constituent_small, structure_memory_small = make_sparse_attention_data(x_train_small, y_train_small)
    x_constituent_large, structure_memory_large = make_sparse_attention_x_data(x_train_large)

    # ---------------------------------------------------------

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
    small_model.save(args.small_model_output_dir + args.small_model_name)

    # ---------------------------------------------------------

    train_large_df = pd.DataFrame(x_constituent_large, columns= select_feature_labels("all"))
    train_large_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_large_df)

    y_pred_train = small_model.predict(train_large_ds)

    x_constituent_large = np.concatenate((x_constituent_large, np.reshape(y_pred_train, (len(y_pred_train),1))), axis = 1)
    x_train_large = np.split(x_constituent_large, np.cumsum(structure_memory_large)[:-1])

    x_heading = generate_X_heading(select_feature_labels("all", include_sparse_attention= True), args.max_constituents)
    
    x_train_large = restrict_nb_constituents(x_train_large, args.max_constituents, padded_value = args.padded_value)
    x_train_large = reshape_X(x_train_large)

    df_train = pd.DataFrame(data = x_train_large, columns = x_heading)
    df_train["class"] = y_train_large

    out_file_name_train = f"jet_images_c{args.max_constituents}_pc{args.positive_class}_SA_train.csv"

    output_file_train = os.path.join(args.output_dir_train, out_file_name_train)
    df_train.to_csv(output_file_train, index = False)

    # ---------------------------------------

    x_test_raw, y_test= load_files_in_directory(args.test_files_dir)
    y_test = [1*(np.argmax(i) == classes.index(args.positive_class)) for i in y_test]

    print("-----------------")
    print("Making testing input for small model")
    x_constituent_test, structure_memory_test = make_sparse_attention_x_data(x_test_raw)
    print("-----------------")

    test_constituent_df = pd.DataFrame(x_constituent_test, columns= select_feature_labels("all"))
    test_constituent_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_constituent_df)

    y_pred_test = small_model.predict(test_constituent_ds)

    x_constituent_test = np.concatenate((x_constituent_test, np.reshape(y_pred_test, (len(y_pred_test),1))), axis = 1)
    x_test = np.split(x_constituent_test, np.cumsum(structure_memory_test)[:-1])

    x_heading = generate_X_heading(select_feature_labels("all", include_sparse_attention= True), args.max_constituents)

    print("Restricting number of constituents.")    
    x_test = restrict_nb_constituents(x_test, args.max_constituents, padded_value = args.padded_value)
    x_test = reshape_X(x_test)

    df_test = pd.DataFrame(data = x_test, columns = x_heading)
    df_test["class"] = y_test

    out_file_name_test = f"jet_images_c{args.max_constituents}_pc{args.positive_class}_SA_test.csv"
    output_file_test = os.path.join(args.output_dir, out_file_name_test)
    df_test.to_csv(output_file_test, index = False)

    return

def make_sparse_attention_data(x_data, y_data):
    boolean_mask = np.any(x_data, axis = 2)
    structure_memory = boolean_mask.sum(axis=1)
    x_constituent_data = x_data[boolean_mask, :]

    y_constituent_data = np.array([], dtype=int)
    for i in range(len(structure_memory)):
        large_class = np.append(large_class, structure_memory[i]*[y_data[i]])

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
        print('\nReading files...')

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
        print('\nAll files read')

    return x_data, y_data

def load_training(train_files_dir, training_split= 0.1, classes= ['g', 'q', 'w', 'z', 't'], 
         positive_class = 't', save: bool = False, output_dir= "/work/aoliver/BDT/sparse_attention/data/"):

    x_train_full, y_train_full = load_files_in_directory(train_files_dir, nb_files = 4)
    y_train_full = [1*(np.argmax(i) == classes.index(positive_class)) for i in y_train_full]
    print('==================')

    print("Splitting training data")
    split_events = int(len(x_train_full)*training_split)

    x_train_small = x_train_full[:split_events]
    x_train_large =  x_train_full[split_events:]

    y_train_small = y_train_full[:split_events]
    y_train_large =  y_train_full[split_events:]
    print("Successfully split training data.")
    print('==================')

    print("Making data for the small model...")
    x_constituent_small, y_constituent_small, structure_memory_small = make_sparse_attention_data(x_train_small, y_train_small)
    x_constituent_large, structure_memory_large = make_sparse_attention_x_data(x_train_large, y_train_large)

    if save:
        print(f"Data successfully made: now saving arrays to '{output_dir}'")
        np.save(os.path.join(output_dir, "x_constituent_small"), x_constituent_small)
        np.save(os.path.join(output_dir, "y_constituent_small"), y_constituent_small)
        np.save(os.path.join(output_dir, "structure_memory_small"), structure_memory_small)

        np.save(os.path.join(output_dir, "x_constituent_large"), x_constituent_large)
        # np.save(os.path.join(output_dir, "y_constituent_large"), y_constituent_large)
        np.save(os.path.join(output_dir, "y_train_large"), y_train_large)
        np.save(os.path.join(output_dir, "structure_memory_large"), structure_memory_large)
    print("Process complete!")

    return x_constituent_small, y_constituent_small, structure_memory_small, x_constituent_large, structure_memory_large, y_train_large

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)