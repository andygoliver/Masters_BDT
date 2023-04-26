import numpy as np
import pandas as pd
import os
import tensorflow_decision_forests as tfdf
from tensorflow.keras.models import load_model 

from train_small_model import select_feature_labels
from load_training import load_files_in_directory, make_sparse_attention_data
from apply_model_to_training import generate_X_heading, reshape_X, restrict_nb_constituents

def main():
    test_files_dir = "/work/aoliver/Data/val/"
    small_model_dir = "models/"
    small_model_name = "small_model_test"

    output_dir = "processed_data/"
    out_file_name = "test.csv"

    max_constituents = 16
    padded_value = -9999.

    classes = ['g', 'q', 'w', 'z', 't']
    positive_class = 't'

    # -----------------------------------------

    x_test_raw, y_test= load_files_in_directory(test_files_dir)
    y_test = [1*(np.argmax(i) == classes.index(positive_class)) for i in y_test]

    print("-----------------")
    print("Making input for small model")
    x_constituent_test, structure_memory = make_sparse_attention_x_data(x_test_raw)
    print("-----------------")

    small_model = load_model(os.path.join(small_model_dir, small_model_name))

    test_constituent_df = pd.DataFrame(x_constituent_test, columns= select_feature_labels("all"))

    test_constituent_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_constituent_df)

    y_pred_test = small_model.predict(test_constituent_ds)

    x_constituent_test = np.concatenate((x_constituent_test, np.reshape(y_pred_test, (len(y_pred_test),1))), axis = 1)
    x_test = np.split(x_constituent_test, np.cumsum(structure_memory)[:-1])

    x_heading = generate_X_heading(select_feature_labels("all", include_sparse_attention= True), max_constituents)

    print("Restricting number of constituents.")    
    x_test = restrict_nb_constituents(x_test, max_constituents, padded_value = padded_value)
    x_test = reshape_X(x_test)

    df_test = pd.DataFrame(data = x_test, columns = x_heading)
    df_test["class"] = y_test

    output_file = os.path.join(output_dir, out_file_name)
    df_test.to_csv(output_file, index = False)
    return df_test

def make_sparse_attention_x_data(x_data):
    boolean_mask = np.any(x_data, axis = 2)
    structure_memory = boolean_mask.sum(axis=1)
    x_constituent_data = x_data[boolean_mask, :]

    return x_constituent_data, structure_memory

if __name__ == "__main__":
    main()