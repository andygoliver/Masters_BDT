import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

import argparse
import time
import os

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
    help="Directory to save the plots in."
)

args = parser.parse_args()

# define classes for the labels (second array is quick fix to bytes problem)
# believe that the model spits the predictions out in alphabetical order
classes = np.array(['g', 'q', 't', 'w', 'z'])
bclasses = np.array(["b'g'", "b'q'", "b't'", "b'w'", "b'z'"])

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
# print(y_pred)
y_test = test_df['class']
# print(y_test)

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
plt.savefig(os.path.join(args.plot_output_dir, args.plot_name))
plt.close()

print('\n=============================================\n')

print('Inputs')
print('------')

print(f'Evaluated model: {args.model_path}')
print(f'Test dataset: {args.data_path_test}')

print('\nOutputs')
print('-------')

print(f'ROC saved as: {args.plot_output_dir + args.plot_name}')

print('\nModel information')
print('--------------------')
    
print("model type:", inspector.model_type())
print("number of trees:", inspector.num_trees())
print("objective:", inspector.objective())
#print("Input features:", inspector.features())

for name, value in evaluation.items():
  print(f"{name}: {value:.4f}")

print('\n=============================================\n')