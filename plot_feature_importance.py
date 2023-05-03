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
    "fig_directory",
    type=str,
    help="Name of directory in which figure will be saved.",
)
parser.add_argument(
    "nb_constituents",
    type=int,
    help="Number of constituents per jet used in the saved model.",
)
parser.add_argument(
    "nb_full_agg_features",
    type=int,
    help="Number of full aggregated features in saved model.",
)
parser.add_argument(
    "--feature_selection",
    type=str,
    default = 'all',
    help="Select features to appear in the plot (should correspond to those present in the model).",
)
parser.add_argument(
    "--agg_feature_nb_constituents",
    type=int,
    default=None,
    nargs = '+',
    help="Specify numbers of constituents used to create aggregated features.",
)
parser.add_argument(
    "--include_nb_constituents",
    action= "store_true",
    help="Toggle inclusion of 'nb_constituents' feature in saved model.",
)
parser.add_argument(
    "--include_sparse_attention",
    action= "store_true",
    help="Toggle inclusion of 'sparse_attention' feature in saved model.",
)

def main(args):
    # importances = ['INV_MEAN_MIN_DEPTH']
    importances  = ['INV_MEAN_MIN_DEPTH', 'NUM_AS_ROOT', 'NUM_NODES', 'SUM_SCORE']

    for importance in importances:
        print(importance)
        plot_importances(args.model_path, importance, args.nb_constituents, nb_full_agg_features = args.nb_full_agg_features, 
                        feature_choice= args.feature_selection, include_nb_constituents=args.include_nb_constituents, 
                        include_sparse_attention= args.include_sparse_attention, agg_feature_nb_constituents= args.agg_feature_nb_constituents,
                        show_fig = False, save_fig = True, fig_name = args.fig_name, fig_directory = args.fig_directory)
        
    inspector_path = os.path.join(args.model_path, "assets")
    inspector = tfdf.inspector.make_inspector(inspector_path)

    print(inspector.variable_importances()['INV_MEAN_MIN_DEPTH'])
        

def get_2d_importances(model_path: str, variable_importance:str, nb_constituents, nb_full_agg_features = 0,
                       agg_feature_nb_constituents = None, include_sparse_attention = False,
                       include_nb_constituents = False, feature_choice: str = 'all') -> np.ndarray:
    """Return the feature importances as a 2D array that can be used to create the seaborn heatmap."""
    feature_list = select_feature_labels(feature_choice, include_sparse_attention)
    full_agg_features = ["mean", "sum", "max"]

    inspector_path = os.path.join(model_path, "assets")
    inspector = tfdf.inspector.make_inspector(inspector_path)

    nb_features = len(feature_list)
    # print(f'features per constituent: {nb_features}')
    # nb_constituents = int(len(inspector.features())/nb_features)
    # print(f'total features: {len(inspector.features())}')
    # print(f'constituents per jet: {nb_constituents}')
    
    feature_importance_dict = {i[0][0]: i[1] for i in inspector.variable_importances()[variable_importance]}

    if agg_feature_nb_constituents is None:
        feature_importances  = np.zeros((nb_features + 1*include_nb_constituents, nb_constituents + nb_full_agg_features + 1*include_nb_constituents))

    else:
        feature_importances  = np.zeros((nb_features + 1*include_nb_constituents, 
                                         nb_constituents + len(agg_feature_nb_constituents)*nb_full_agg_features + 1*include_nb_constituents))     
    
    for feature_index in range(nb_features):
        for constituent_index in range(nb_constituents):
            try:
                feature_importances[feature_index][constituent_index] = feature_importance_dict[f'c{constituent_index}_{feature_list[feature_index]}']
                # print(f'c{constituent_index}_{feature_list[feature_index]} was found')

            except:
                # print(f'c{constituent_index}_{feature_list[feature_index]} was not found')
                continue

        if agg_feature_nb_constituents is None:
            for full_agg_index in range(nb_full_agg_features):
                try:
                    feature_importances[feature_index][full_agg_index + nb_constituents] = feature_importance_dict[f'{feature_list[feature_index]}_{full_agg_features[full_agg_index]}']
                    # print(f'{feature_list[feature_index]}_{full_agg_features[full_agg_index]} was found')

                except:
                    # print(f'{feature_list[feature_index]}_{full_agg_features[full_agg_index]} was not found')
                    continue

        else: 
            for full_agg_index in range(nb_full_agg_features):
                for agg_constituent_index in range(len(agg_feature_nb_constituents)):
                    n = agg_feature_nb_constituents[agg_constituent_index]
                    if n == 150:
                        flag = ''
                    else:
                        flag = f'_c{n}'
                
                    try:
                        x = feature_index
                        y = full_agg_index+ nb_full_agg_features*agg_constituent_index + nb_constituents
                        feature_importances[x][y] = feature_importance_dict[f'{feature_list[feature_index]}_{full_agg_features[full_agg_index]}{flag}']
                        # print(f'{feature_list[feature_index]}_{full_agg_features[full_agg_index]}{flag} was found')

                    except:
                        # print(f'{feature_list[feature_index]}_{full_agg_features[full_agg_index]}{flag} was not found')
                        continue

    if include_nb_constituents:            
        try:
            feature_importances[-1][-1] = feature_importance_dict['nb_constituents']
            print(f'nb_constituents was found')

        except:
            print(f'nb_constituents was not found')
    
    return feature_importances

def plot_importances(model_path: str, variable_importance:str, nb_constituents:int, nb_full_agg_features:int = 0, include_nb_constituents:bool = False,
                     include_sparse_attention= False, agg_feature_nb_constituents = None, feature_choice: str = 'jedinet', cmap: str = "YlGnBu",
                     save_fig: bool = False, fig_name:str = '', fig_directory:str = 'Plots', show_fig:bool = False):
    
    feature_importances = get_2d_importances(model_path, variable_importance, nb_constituents, nb_full_agg_features=nb_full_agg_features, 
                                             agg_feature_nb_constituents= agg_feature_nb_constituents, include_sparse_attention=include_sparse_attention,
                                             include_nb_constituents=include_nb_constituents,  feature_choice = feature_choice)
    full_agg_features = ["mean", "sum", "max"]

    xlabels = np.arange(nb_constituents)

    if agg_feature_nb_constituents == None:
        for i in range(nb_full_agg_features):
            xlabels = np.append(xlabels, full_agg_features[i])

    else:
        for n in agg_feature_nb_constituents:
            for full_agg_feature in full_agg_features:
                xlabels = np.append(xlabels, f"{full_agg_feature}_c{n}")

    if include_nb_constituents:
        xlabels = np.append(xlabels, 'nb_constituents')

    # use fisize (18.5,5) for c50 and (55,5) for c150, width ~3/8*nb_constituents 
    # plt.subplots(figsize=(18.5,5))
    plt.subplots(figsize=(10,5))
    ax = sns.heatmap(feature_importances, linewidth=0.5, cmap=cmap, yticklabels = select_feature_labels(feature_choice, include_sparse_attention), 
                     xticklabels = xlabels, square = True, mask = feature_importances == 0)
    ax.set(ylabel = 'Feature', title=f'{variable_importance} feature importance')

    if agg_feature_nb_constituents == None:
        nb_extra_labels = nb_full_agg_features + 1*include_nb_constituents
    else:
        nb_extra_labels = nb_full_agg_features*len(agg_feature_nb_constituents) + 1*include_nb_constituents
    
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

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)