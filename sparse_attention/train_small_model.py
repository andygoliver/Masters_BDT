import numpy as np
import os
import pandas as pd

import tensorflow_decision_forests as tfdf

def main():
    input_dir = "data/"
    BDT_hyperparameters = {'num_trees':60,
                           'shrinkage':0.2,
                           'subsample':0.3,
                           'use_hessian_gain':True,
                           'growing_strategy':'BEST_FIRST_GLOBAL',
                           'max_depth':-1,
                           'max_num_nodes':32,
                           'num_threads': 64
                           }
    small_model_output_dir = "models/"
    small_model_name = "small_model_test"

    print("Loading data")
    x_constituent_small = np.load(os.path.join(input_dir, "x_constituent_small.npy"), "r")
    y_constituent_small = np.load(os.path.join(input_dir, "y_constituent_small.npy"), "r")

    print("Data loaded: now making dataframes")
    train_small_df = pd.DataFrame(x_constituent_small, columns= select_feature_labels("all"))
    train_small_df["class"] = y_constituent_small

    train_small_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_small_df, label="class")

    small_model = tfdf.keras.GradientBoostedTreesModel(**BDT_hyperparameters)

    small_model.fit(train_small_ds, verbose = 0)
    small_model.save(small_model_output_dir + small_model_name)

    inspector = small_model.make_inspector()

    print_model_information(input_dir, small_model_output_dir, small_model_name, BDT_hyperparameters, inspector)
    return 

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

def print_model_information(data_path_train, model_output_dir, model_name, BDT_hyperparameters, inspector, training_time = None):
    print('\n=============================================\n')

    print('Inputs')
    print('------')

    print(f'Training data: {data_path_train}')

    print('\nOutputs')
    print('-------')

    print(f'Model saved as: {model_output_dir + model_name}')

    """
    if args.plot_ROC:
        '''
        print(f'Plot output folder: {plot_output_dir}')
        print(f'Plot name: {plot_name}')
        '''
        print(f'ROC saved as: {args.plot_output_dir + args.plot_name}')
        """

    print('\nModel hyperparameters')
    print('---------------------')

    for hyperparameter in BDT_hyperparameters:
        print(f'{hyperparameter}: {BDT_hyperparameters[hyperparameter]}')

    print('\nTraining information')
    print('--------------------')
        
    print("model type:", inspector.model_type())
    print("number of trees:", inspector.num_trees())
    print("objective:", inspector.objective())

    if training_time != None:
        print(f"training time: {training_time:.2f}s")

    print('\n=============================================\n')

if __name__ == "__main__":
    main()