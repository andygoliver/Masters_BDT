import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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
    "nb_constituents",
    type=int,
    help="Number of constituents per jet used in the saved model.",
)
parser.add_argument(
    "--nb_full_agg_features",
    type=int,
    default = 0,
    help="Number of full aggregated features in saved model.",
)
parser.add_argument(
    "--include_nb_constituents",
    type=bool,
    default = False,
    help="Toggle inclusion of 'nb_constituents' feature in saved model.",
)
parser.add_argument(
    "--fig_directory",
    type=str,
    default = "Plots/Feat_imp",
    help="Name of directory in which figure will be saved.",
)

def get_2d_importances(model_path: str, variable_importance:str, nb_constituents, nb_full_agg_features = 0, 
                       include_nb_constituents = False, feature_choice: str = 'jedinet') -> np.ndarray:
    """Return the feature importances as a 2D array that can be used to create the seaborn heatmap."""
    feature_list = select_feature_labels(feature_choice)
    full_agg_features = ["mean", "sum"]

    inspector_path = os.path.join(model_path, "assets")
    inspector = tfdf.inspector.make_inspector(inspector_path)

    nb_features = len(feature_list)
    # print(f'features per constituent: {nb_features}')
    # nb_constituents = int(len(inspector.features())/nb_features)
    # print(f'total features: {len(inspector.features())}')
    # print(f'constituents per jet: {nb_constituents}')
    
    feature_importance_dict = {i[0][0]: i[1] for i in inspector.variable_importances()[variable_importance]}

    feature_importances  = np.zeros((nb_features + 1*include_nb_constituents, nb_constituents + nb_full_agg_features + 1*include_nb_constituents))
    for feature_index in range(nb_features):
        for constituent_index in range(nb_constituents):
            try:
                feature_importances[feature_index][constituent_index] = feature_importance_dict[f'c{constituent_index}_{feature_list[feature_index]}']
                # print(f'c{constituent_index}_{feature_list[feature_index]} was found')

            except:
                # print(f'c{constituent_index}_{feature_list[feature_index]} was not found')
                continue

        for full_agg_index in range(nb_full_agg_features):
            try:
                feature_importances[feature_index][full_agg_index + nb_constituents] = feature_importance_dict[f'{feature_list[feature_index]}_{full_agg_features[full_agg_index]}']
                # print(f'{feature_list[feature_index]}_{full_agg_features[full_agg_index]} was found')

            except:
                # print(f'{feature_list[feature_index]}_{full_agg_features[full_agg_index]} was not found')
                continue
    if include_nb_constituents:            
        try:
            feature_importances[-1][-1] = feature_importance_dict['nb_constituents']
            print(f'nb_constituents was found')

        except:
            print(f'nb_constituents was not found')
    
    return feature_importances

def plot_importances(model_path: str, variable_importance:str, nb_constituents:int, nb_full_agg_features:int = 0, include_nb_constituents:bool = False,
                     feature_choice: str = 'jedinet', cmap: str = "YlGnBu",
                     save_fig: bool = False, fig_name:str = '', fig_directory:str = 'Plots', show_fig:bool = False):
    
    feature_importances = get_2d_importances(model_path, variable_importance, nb_constituents, nb_full_agg_features=nb_full_agg_features, 
                                             include_nb_constituents=include_nb_constituents,  feature_choice = feature_choice)
    full_agg_features = ["mean", "sum"]

    xlabels = np.arange(nb_constituents)

    for i in range(nb_full_agg_features):
        xlabels = np.append(xlabels, full_agg_features[i])

    if include_nb_constituents:
        xlabels = np.append(xlabels, 'nb_constituents')

    # use fisize (18.5,5) for c50 and (55,5) for c150, width ~3/8*nb_constituents 
    # plt.subplots(figsize=(18.5,5))
    ax = sns.heatmap(feature_importances, linewidth=0.5, cmap=cmap, yticklabels = select_feature_labels(feature_choice), 
                     xticklabels = xlabels, square = True, mask = feature_importances == 0)
    ax.set(ylabel = 'Feature', title=f'{variable_importance} feature importance')

    nb_extra_labels = nb_full_agg_features + 1*include_nb_constituents
    plt.setp(ax.get_xticklabels()[-nb_extra_labels:], rotation=60, ha="right",
             rotation_mode="anchor")
    plt.setp(ax.get_xticklabels()[:-nb_extra_labels], rotation=0)

    plt.tight_layout()
    if save_fig:
        savepath = os.path.join(fig_directory, f"{variable_importance}_{fig_name}.pdf")
        plt.savefig(savepath)
    if show_fig:
        plt.show()
    
    plt.close()

args = parser.parse_args()
# importances = ['INV_MEAN_MIN_DEPTH']
importances  = ['INV_MEAN_MIN_DEPTH', 'NUM_AS_ROOT', 'NUM_NODES', 'SUM_SCORE']

for importance in importances:
    print(importance)
    plot_importances(args.model_path, importance, args.nb_constituents, nb_full_agg_features = args.nb_full_agg_features, 
                     include_nb_constituents=args.include_nb_constituents, 
                     show_fig = False, save_fig = True, fig_name = args.fig_name, fig_directory = args.fig_directory)