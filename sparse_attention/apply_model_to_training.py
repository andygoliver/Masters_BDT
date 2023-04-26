import numpy as np
import pandas as pd
import os
import tensorflow_decision_forests as tfdf
from tensorflow.keras.models import load_model 

from train_small_model import select_feature_labels

def main():
    input_dir = "data/"
    small_model_dir = "models/"
    small_model_name = "small_model_test"

    output_dir = "processed_data/"
    out_file_name = "train.csv"

    max_constituents = 16
    padded_value = -9999.

    print("Loading data")
    x_constituent_large = np.load(os.path.join(input_dir, "x_constituent_small.npy"), "r")
    y_constituent_large = np.load(os.path.join(input_dir, "y_constituent_small.npy"), "r")
    y_train_large = np.load(os.path.join(input_dir, "y_train_large.npy"), "r")
    structure_memory_large = np.load(os.path.join(input_dir, "structure_memory_large.npy"),"r")

    small_model = load_model(os.path.join(small_model_dir, small_model_name))

    train_large_df = pd.DataFrame(x_constituent_large, columns= select_feature_labels("all"))
    train_large_df["class"] = y_constituent_large

    train_large_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_large_df, label="class")

    y_pred_train = small_model.predict(train_large_ds)

    x_constituent_large = np.concatenate((x_constituent_large, np.reshape(y_pred_train, (len(y_pred_train),1))), axis = 1)
    x_train_large = np.split(x_constituent_large, np.cumsum(structure_memory_large)[:-1])

    x_heading = generate_X_heading(select_feature_labels("all", include_sparse_attention= True), max_constituents)
    
    x_train_large = restrict_nb_constituents(x_train_large, max_constituents, padded_value = padded_value)
    x_train_large = reshape_X(x_train_large)

    df_train = pd.DataFrame(data = x_train_large, columns = x_heading)
    df_train["class"] = y_train_large

    output_file = os.path.join(output_dir, out_file_name)
    df_train.to_csv(output_file, index = False)
    return df_train

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

if __name__ == "__main__":
    main()