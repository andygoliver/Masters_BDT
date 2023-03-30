import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import argparse

import numpy as np
import matplotlib.pyplot as plt
import tensorflow_decision_forests as tfdf
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import time

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--input_folder",
    type=str,
    required=True,
    help="Path to the input data.",
)
parser.add_argument(
    "--filename",
    type=str,
    required=True,
    help="Name of the middle part of training/testing files (using Patrick's convention).",
)
parser.add_argument(
    "--train_flag",
    type=str,
    default="_train",
    help="Flag for training data.",
)
parser.add_argument(
    "--test_flag",
    type=str,
    default="_test",
    help="Flag for testing data.",
)
parser.add_argument(
    "--BDT_hyperparameters",
    type=dict,
    default={'num_trees':5,
             'shrinkage':0.2,
             'subsample':0.3,
             'use_hessian_gain':True,
             'growing_strategy':'BEST_FIRST_GLOBAL',
             'max_depth':-1,
             'max_num_nodes':32
            },
    help="Hyperparameters of the BDT. Add as a dictionary.",
)
parser.add_argument(
    "--model_output_dir",
    type=str,
    default="Models/",
    help="Directory in which to save the trained model.",
)
parser.add_argument(
    "--model_name",
    type=str,
    default="testy_test",
    help="Name of output model.",
)
parser.add_argument(
    "--plot_ROC",
    type=bool,
    default=False,
    help="Toggles ROC plotting on or off (default off).",
)
parser.add_argument(
    "--plot_output_dir",
    type=str,
    default="Plots/",
    help="Directory in which to save the ROC plot.",
)
parser.add_argument(
    "--plot_name",
    type=str,
    default="test.png",
    help="Name of output ROC plot.",
)
args = parser.parse_args()

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

filename_X_train = args.input_folder + 'x_' + args.filename + args.train_flag + '.npy'
filename_X_test = args.input_folder + 'x_' + args.filename + args.test_flag + '.npy'
filename_y_train = args.input_folder + 'y_' + args.filename + args.train_flag + '.npy'
filename_y_test = args.input_folder + 'y_' + args.filename + args.test_flag + '.npy'

#load input data 
X_train = np.load(filename_X_train)
X_test = np.load(filename_X_test)
y_train = np.load(filename_y_train).astype(int)
y_test = np.load(filename_y_test).astype(int)

#reshape X arrays
X_train_reshaped = reshape_X(X_train)
X_test_reshaped = reshape_X(X_test)

#specify classes (used to plot ROC curve and create OHE) and features (no use yet)
classes = np.array(['g', 'q', 'w', 'z', 't'])
andre_feature_labels = np.array(["p_T", "\\eta^\\mathrm{rel}", "\\phi^\\mathrm{rel}"])

#create a label encoder with the desired classes 
le = LabelEncoder().fit(classes)

#create a one-hot encoder using this label and the classes
#applying le.transform(classes) gives labels to the class data
#this means that we get an ordered list integer labels, since we used the classes array to create the encoder
#array needs to be reshaped as the .fit() function requires a 2D array
ohe = OneHotEncoder().fit(le.transform(classes).reshape(-1,1))

#use this one-hot encoder to go from the one-hot encoded y to a labelled y for both the training and testing y
y_train_val = ohe.inverse_transform(y_train.astype(int))
y_test_val = ohe.inverse_transform(y_test.astype(int))

'''
BDT_hyperparameters = {'num_trees':5,
                      'shrinkage':0.2,
                      'subsample':0.3,
                      'use_hessian_gain':True,
                      'growing_strategy':'BEST_FIRST_GLOBAL',
                      'max_depth':-1,
                      'max_num_nodes':32
                     }
'''

model = tfdf.keras.GradientBoostedTreesModel(**args.BDT_hyperparameters)

starting_time = time.time()
model.fit(X_train_reshaped, y_train_val, verbose = 0)
training_time = time.time() - starting_time

'''
model_output_dir = 'Models/'
model_name = 'model_test'
'''

model.save(args.model_output_dir + args.model_name)

model.compile(metrics=["accuracy"])
evaluation = model.evaluate(X_test_reshaped, y_test_val, return_dict=True)
inspector = model.make_inspector()

plot_ROC = True
'''
plot_output_dir = 'Plots/'
plot_name = 'test.png'
'''

if args.plot_ROC:
    print('\nObtaining data for plots...\n')

    y_pred = model.predict(X_test_reshaped)

    from sklearn.metrics import roc_curve, auc, accuracy_score

    #print(y_test)
    #print('\n-----------\n')
    #print(y_pred)

    fpr = {}
    tpr = {}
    aucs = {}

    for i, label in enumerate(classes):
        #print(f'label = {label}')
        #print(f'i = {i}')
        #print(f'y_true = {y_test[:,i]}')
        #print(f'y_pred = {y_pred[:,i]}')
        fpr[label], tpr[label], threshold = roc_curve(y_test[:,i],y_pred[:,i])
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
    plt.figtext(0.25, 0.90,'Initial test',fontweight='bold', wrap=True, horizontalalignment='right', fontsize=14)
    plt.savefig(args.plot_output_dir + args.plot_name)
    #plt.savefig('Plots/test.png')
    plt.close()
    #plt.show()

print('\n=============================================\n')

print('Inputs')
print('------')

print(f'X training data: {filename_X_train}')
print(f'y training data: {filename_y_train}')
print(f'X testing data: {filename_X_test}')
print(f'y testing data: {filename_y_test}')

print('\nOutputs')
print('-------')

'''
print(f'Model output folder: {model_output_dir}')
print(f'Model name: {model_name}')
'''
print(f'Model saved as: {args.model_output_dir + args.model_name}')

if args.plot_ROC:
    '''
    print(f'Plot output folder: {plot_output_dir}')
    print(f'Plot name: {plot_name}')
    '''
    print(f'ROC saved as: {args.plot_output_dir + args.plot_name}')

print('\nModel hyperparameters')
print('---------------------')

for hyperparameter in args.BDT_hyperparameters:
    print(f'{hyperparameter}: {args.BDT_hyperparameters[hyperparameter]}')

print('\nTraining information')
print('--------------------')
    
print("model type:", inspector.model_type())
print("number of trees:", inspector.num_trees())
print("objective:", inspector.objective())
#print("Input features:", inspector.features())

for name, value in evaluation.items():
  print(f"{name}: {value:.4f}")

print(f"training time: {training_time:.2f}s")

print('\n=============================================\n')
