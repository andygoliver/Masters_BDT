import os
import sys
import argparse

import h5py
import numpy as np

import pandas as pd

from prepare_data_BDT 

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "data_file_dir",
    type=str,
    help="Path to directories with .h5 files containing jet image data.",
)
parser.add_argument(
    "output_dir", type=str, help="Directory to save the processed data in."
)

def main(args):
    x_data, y_data = load_all_files_in_directory(args.data_file_dir)