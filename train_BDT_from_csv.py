import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import argparse

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import tensorflow_decision_forests as tfdf
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import time

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--data_path_train",
    type=str,
    required=True,
    help="Path to the training data file to process.",
)
'''
parser.add_argument(
    "--data_path_test",
    type=str,
    required=True,
    help="Path to the test data file to process.",
)
'''
parser.add_argument(
    "--nb_jets",
    type=int,
    default=None,
    help="Number of events to load in.",
)
parser.add_argument(
    "--BDT_hyperparameters",
    type=dict,
    default={'num_trees':60,
             'shrinkage':0.2,
             'subsample':0.3,
             'use_hessian_gain':True,
             'growing_strategy':'BEST_FIRST_GLOBAL',
             'max_depth':-1,
             'max_num_nodes':32,
             'num_threads': 64
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

def load_tfds(data_path: str, nrows_:int = None, name:str = None, label_: str = 'class'):
   start_time = time.time()
   df = pd.read_csv(data_path, nrows = nrows_)
   ds = tfdf.keras.pd_dataframe_to_tf_dataset(df, label=label_)
   load_time = time.time()-start_time

   if name is not None:
      print(f'Loaded {name} sample of {len(df)} jets in {load_time:.3f}s')

   return ds

args = parser.parse_args()

print('\n=============================================\n')

# Load in the data
train_ds = load_tfds(args.data_path_train, nrows_ = args.nb_jets, name = 'training')
# test_ds= load_tfds(args.data_path_test, name = 'testing')

print('\n=============================================\n')


# define the model
model = tfdf.keras.GradientBoostedTreesModel(**args.BDT_hyperparameters)

#train the model
starting_time = time.time()
model.fit(train_ds, verbose = 0)
training_time = time.time() - starting_time

# save the model
model.save(args.model_output_dir + args.model_name)

# evaluate the model 
'''
model.compile(metrics=["accuracy"])
evaluation = model.evaluate(train_ds, return_dict=True)
'''
inspector = model.make_inspector()

print('\n=============================================\n')

print('Inputs')
print('------')

print(f'Training data: {args.data_path_train}')
# print(f'Testing data: {args.data_path_test}')

print('\nOutputs')
print('-------')

print(f'Model saved as: {args.model_output_dir + args.model_name}')

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

for hyperparameter in args.BDT_hyperparameters:
    print(f'{hyperparameter}: {args.BDT_hyperparameters[hyperparameter]}')

print('\nTraining information')
print('--------------------')
    
print("model type:", inspector.model_type())
print("number of trees:", inspector.num_trees())
print("objective:", inspector.objective())
#print("Input features:", inspector.features())

'''
for name, value in evaluation.items():
  print(f"{name}: {value:.4f}")
'''

print(f"training time: {training_time:.2f}s")

print('\n=============================================\n')
