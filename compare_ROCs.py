import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

import argparse
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow_decision_forests as tfdf
from tensorflow.keras.models import load_model

from sklearn.metrics import roc_curve, auc

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "model_1_path",
    type=str,
    help="Path to the first model to process.",
)
parser.add_argument(
    "model_1_label",
    type=str,
    help="Label of the first model to process.",
)
parser.add_argument(
    "model_2_path",
    type=str,
    help="Path to the second model to process.",
)
parser.add_argument(
    "model_2_label",
    type=str,
    help="Label of the second model to process.",
)
parser.add_argument(
    "data_path_test_1",
    type=str,
    help="Path to the first test data file to process.",
)
parser.add_argument(
    "data_path_test_2",
    type=str,
    help="Path to the second test data file to process.",
)
parser.add_argument(
    "plot_output_dir", 
    type=str, 
    help="Directory to save the plot in."
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

args = parser.parse_args()

test_starting_time = time.time()
test_df_1 = pd.read_csv(args.data_path_test_1, nrows = None)
test_load_time = time.time()-test_starting_time
print(f'Loaded testing sample 1 of {len(test_df_1)} jets in {test_load_time:.3f}s')

test_ds_1 = tfdf.keras.pd_dataframe_to_tf_dataset(test_df_1, label='class')

test_starting_time = time.time()
test_df_2 = pd.read_csv(args.data_path_test_2, nrows = None)
test_load_time = time.time()-test_starting_time
print(f'Loaded testing sample 2 of {len(test_df_2)} jets in {test_load_time:.3f}s')

test_ds_2 = tfdf.keras.pd_dataframe_to_tf_dataset(test_df_2, label='class')

model_1 = load_model(args.model_1_path)
model_2 = load_model(args.model_2_path)

model_1.compile(metrics=["accuracy"])
evaluation_1 = model_1.evaluate(test_ds_1, return_dict=True)

model_2.compile(metrics=["accuracy"])
evaluation_2 = model_2.evaluate(test_ds_2, return_dict=True)

inspector_1_path = os.path.join(args.model_1_path, "assets")
inspector_1 = tfdf.inspector.make_inspector(inspector_1_path)

inspector_2_path = os.path.join(args.model_2_path, "assets")
inspector_2 = tfdf.inspector.make_inspector(inspector_2_path)

y_pred_1 = model_1.predict(test_ds_1)
y_pred_2 = model_2.predict(test_ds_2)
y_test_1 = test_df_1['class']
y_test_2 = test_df_2['class']

fpr_1, tpr_1, threshold = roc_curve(y_test_1,y_pred_1)
auc_1 = auc(fpr_1,tpr_1)
fpr_2, tpr_2, threshold = roc_curve(y_test_2,y_pred_2)
auc_2 = auc(fpr_2,tpr_2)
#print('\n--------------------------------\n')

fig, ax = plt.subplots(figsize=(9, 9))
plt.plot(tpr_1,fpr_1, label = f't tagger {args.model_1_label}, AUC = {auc_1*100:.1f}%')    
plt.plot(tpr_2,fpr_2, label = f't tagger {args.model_2_label}, AUC = {auc_2*100:.1f}%')  
plt.semilogy()
plt.xlabel("Signal Efficiency")
plt.ylabel("Background Efficiency")
plt.ylim(0.001,1)
plt.grid(True)
plt.legend(loc='upper left')
plt.figtext(0.25, 0.90,f"Accuracy_{args.model_1_label} = {evaluation_1['accuracy']:.4f}, Accuracy_{args.model_2_label} = {evaluation_2['accuracy']:.4f}",fontweight='bold', wrap=True, horizontalalignment='right', fontsize=14)
plt.savefig(os.path.join(args.plot_output_dir, args.plot_name))
plt.close()