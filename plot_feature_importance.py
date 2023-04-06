import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow_decision_forests as tfdf

from prepare_data_BDT import select_feature_labels


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "model_path",
    type=str,
    help="Path to the model to process.",
)
parser.add_argument(
    "fig_name",
    type=str,
    help="Name of saved figure.",
)
parser.add_argument(
    "--fig_directory",
    type=str,
    default = "Plots/Feat_imp",
    help="Name of directory in which figure will be saved.",
)

def get_2d_importances(model_path: str, variable_importance:str, feature_choice: str = 'jedinet') -> np.ndarray:
    feature_list = select_feature_labels(feature_choice)

    inspector_path = os.path.join(model_path, "assets")
    inspector = tfdf.inspector.make_inspector(inspector_path)

    nb_features = len(feature_list)
    # print(f'features per constituent: {nb_features}')
    nb_constituents = int(len(inspector.features())/nb_features)
    # print(f'total features: {len(inspector.features())}')
    # print(f'constituents per jet: {nb_constituents}')
    
    feature_importance_dict = {i[0][0]: i[1] for i in inspector.variable_importances()[variable_importance]}

    feature_importances  = np.zeros((nb_features, nb_constituents))
    for feature_index in range(nb_features):
        for constituent_index in range(nb_constituents):
            try:
                feature_importances[feature_index][constituent_index] = feature_importance_dict[f'c{constituent_index}_{feature_list[feature_index]}']
                # print(f'c{constituent_index}_{feature_list[feature_index]} was found')

            except:
                # print(f'c{constituent_index}_{feature_list[feature_index]} was not found')
                continue
    
    return feature_importances

def plot_importances(model_path: str, variable_importance:str, feature_choice: str = 'jedinet', cmap: str = "YlGnBu",
                     save_fig: bool = False, fig_name:str = '', fig_directory:str = 'Plots', show_fig:bool = False):
    feature_importances = get_2d_importances(model_path, variable_importance, feature_choice = feature_choice)
    
    # use fisize (18.5,5) for c50 and (55,5) for c150, width ~3/8*nb_constituents 
    plt.subplots(figsize=(18.5,5))
    ax = sns.heatmap(feature_importances, linewidth=0.5, cmap=cmap, yticklabels = select_feature_labels(feature_choice), square = True, vmin = 0.083)
    ax.set(xlabel='Constituent', ylabel = 'Feature', title=f'{variable_importance} feature importance')
    plt.tight_layout()
    if save_fig:
        savepath = os.path.join(fig_directory, f"{variable_importance}_{fig_name}.pdf")
        plt.savefig(savepath)
    if show_fig:
        plt.show()
    
    plt.close()

args = parser.parse_args()
importances = ['INV_MEAN_MIN_DEPTH']
# importances  = ['INV_MEAN_MIN_DEPTH', 'NUM_AS_ROOT', 'NUM_NODES', 'SUM_SCORE']

for importance in importances:
    print(importance)
    plot_importances(args.model_path, 
                     importance, show_fig = False,
                     save_fig = True, fig_name = args.fig_name, fig_directory = args.fig_directory)