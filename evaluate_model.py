import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

import pickle
import argparse
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow_decision_forests as tfdf
from tensorflow.keras.models import load_model

from sklearn.metrics import roc_curve, auc

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "model_path",
    type=str,
    help="Path to the model to process.",
)
parser.add_argument(
    "data_path_test",
    type=str,
    help="Path to the test data file to process.",
)
parser.add_argument(
    "plot_output_dir", 
    type=str, 
    help="Directory to save the plots in."
)
parser.add_argument(
    "plot_name", 
    type=str, 
    help="Name of the saved plot."
)
parser.add_argument(
    "--classification_type", 
    type=str,
    default='binary',
    choices = ['multiclass', 'binary'], 
    help="Type of classification performed by the model."
)

def main(args):
    test_starting_time = time.time()
    test_df = pd.read_csv(args.data_path_test, nrows = None)
    test_load_time = time.time()-test_starting_time
    print(f'Loaded testing sample of {len(test_df)} jets in {test_load_time:.3f}s')

    test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_df, label='class')
    model = load_model(args.model_path)

    model.compile(metrics=["accuracy"])
    evaluation = model.evaluate(test_ds, return_dict=True)

    inspector_path = os.path.join(args.model_path, "assets")
    inspector = tfdf.inspector.make_inspector(inspector_path)

    y_pred = model.predict(test_ds)
    y_test = test_df['class']

    if args.classification_type == 'multiclass':
        make_multiclass_ROC(y_pred, y_test, args.plot_name, args.plot_output_dir, evaluation)

    else:
        fpr, tpr, auc_, threshold = make_binary_ROC(y_pred, y_test, args.plot_name, args.plot_output_dir, evaluation)

        data_output_directory = os.path.join(args.plot_output_dir, "Evaluation_data/")

        make_and_save_dictionary(evaluation, inspector, args, auc_, data_output_directory)
        np.save(os.path.join(data_output_directory, args.plot_name + "_fpr"), fpr)
        np.save(os.path.join(data_output_directory, args.plot_name + "_tpr"), tpr)

    print_evaluation_information(evaluation, inspector, args)

def make_binary_ROC(y_pred, y_test, plot_name, plot_output_dir, evaluation):
    fpr, tpr, threshold = roc_curve(y_test,y_pred)
    auc_ = auc(fpr,tpr)
    #print('\n--------------------------------\n')

    fig, ax = plt.subplots(figsize=(9, 9))
    plt.plot(tpr,fpr, label = f't tagger, AUC = {auc_*100:.1f}%')    
    plt.semilogy()
    plt.xlabel("Signal Efficiency")
    plt.ylabel("Background Efficiency")
    plt.ylim(0.001,1)
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.figtext(0.25, 0.90,f"Accuracy = {evaluation['accuracy']:.4f}",fontweight='bold', wrap=True, horizontalalignment='right', fontsize=14)
    plt.savefig(os.path.join(plot_output_dir, plot_name + ".pdf"))
    plt.close()

    return fpr, tpr, auc_, threshold

def make_multiclass_ROC(y_pred, y_test, plot_name, plot_output_dir, evaluation):
    # define classes for the labels (second array is quick fix to bytes problem)
    # believe that the model spits the predictions out in alphabetical order
    classes = np.array(['g', 'q', 't', 'w', 'z'])
    bclasses = np.array(["b'g'", "b'q'", "b't'", "b'w'", "b'z'"])

    fpr = {}
    tpr = {}
    aucs = {}

    for i, label in enumerate(classes):
        print(f'{[int(y) for y in y_test == bclasses[i]][:5]}, {[int(y) for y in y_test == bclasses[i]][-5:]}')
        print(f'{y_pred[:,i][:5]}, {y_pred[:,i][-5:]}')
        fpr[label], tpr[label], threshold = roc_curve([int(y) for y in y_test == bclasses[i]],y_pred[:,i])
        aucs[label] = auc(fpr[label],tpr[label])
        #print('\n--------------------------------\n')

    fig, ax = plt.subplots(figsize=(9, 9))
    for i, label in enumerate(classes):
        plt.plot(tpr[label],fpr[label], label = f'{label} tagger, AUC = {aucs[label]*100:.1f}%')    
    plt.semilogy()
    plt.xlabel("Signal Efficiency")
    plt.ylabel("Background Efficiency")
    plt.ylim(0.001,1)
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.figtext(0.25, 0.90,f"Accuracy = {evaluation['accuracy']:.4f}",fontweight='bold', wrap=True, horizontalalignment='right', fontsize=14)
    plt.savefig(os.path.join(plot_output_dir, plot_name + ".pdf"))
    plt.close()

def make_and_save_dictionary(evaluation, inspector, args, auc_, data_output_directory):
    evaluation_dict = {}

    for name, value in evaluation.items():
        evaluation_dict[name] = value

    evaluation_dict['auc'] = auc_

    evaluation_dict["model type"] = inspector.model_type()
    evaluation_dict["number of trees"] = inspector.num_trees()
    evaluation_dict["objective"] =  inspector.objective()

    evaluation_dict["Evaluated model"] = args.model_path
    evaluation_dict["Test dataset"] = args.data_path_test

    evaluation_dict["ROC file name"] = args.plot_output_dir + args.plot_name + ".pdf"

    saved_dictionary_name = args.plot_name + ".pkl"

    with open(os.path.join(data_output_directory, saved_dictionary_name), "wb") as f:
        pickle.dump(evaluation_dict, f)

    return

def print_evaluation_information(evaluation, inspector, args):
    print('\n=============================================\n')

    print('Inputs')
    print('------')

    print(f'Evaluated model: {args.model_path}')
    print(f'Test dataset: {args.data_path_test}')

    print('\nOutputs')
    print('-------')

    print(f'ROC saved as: {args.plot_output_dir + args.plot_name + ".pdf"}')

    print('\nModel information')
    print('--------------------')
        
    print("model type:", inspector.model_type())
    print("number of trees:", inspector.num_trees())
    print("objective:", inspector.objective())
    #print("Input features:", inspector.features())

    for name, value in evaluation.items():
        print(f"{name}: {value:.4f}")

    print('\n=============================================\n')

    return

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)