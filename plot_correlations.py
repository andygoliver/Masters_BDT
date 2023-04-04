import numpy as np
import pandas as pd

import time
import sys

import seaborn as sns
import matplotlib.pyplot as plt

from prepare_data_BDT import select_feature_labels  

# load training data

train_starting_time = time.time()
train_df = pd.read_csv('/work/aoliver/Data/jet_images_c16_pt0_jedinet_sort_hpT_pcNone_train.csv', nrows = None)
train_load_time = time.time()-train_starting_time
print(f'Loaded training sample of {len(train_df)} jets in {train_load_time:.3f}s')

# load testing data

'''
test_starting_time = time.time()
test_df = pd.read_csv('../Data/jet_images_c16_pt0_jedinet_sort_hpT_pcNone_test.csv', nrows = None)
test_load_time = time.time()-test_starting_time
print(f'Loaded testing sample of {len(test_df)} jets in {test_load_time:.3f}s')
'''

corr_start_time = time.time()
corr_mtx = train_df.corr(numeric_only=True)
corr_total_time = time.time() - corr_start_time
print(f'\nCreated correlation matrix in {corr_total_time:.3f}s\n')

corr_mtx_per_const = np.zeros((16,16,16))
feature_list = select_feature_labels('jedinet')

for i in range(16):
    features = [f'c{i}_{feature}' for feature in feature_list]
    corr_mtx_per_const[i] = corr_mtx.loc[features][features].to_numpy()

    ax = sns.heatmap(corr_mtx.loc[features][features], linewidth=0.5, vmin = -1, vmax = 1,cmap = 'magma', yticklabels = feature_list, xticklabels = feature_list, square = True)
    ax.set(title=f'Correlations for constituent {i}')
    plt.tight_layout()
    plt.savefig(f'Plots/Correlations/Correlations_{i}.pdf')
    plt.close()

    sys.stdout.write('\r')
    sys.stdout.write(f"Made plot [{i+1}/{16}]")
    sys.stdout.flush()

print('\n')

avg_corr_mtx = np.mean(corr_mtx_per_const, axis = 0)

ax = sns.heatmap(avg_corr_mtx, linewidth=0.5, vmin = -1, vmax = 1, cmap = 'magma', yticklabels = feature_list, xticklabels = feature_list, square = True)
ax.set(title=f'Average correlations per constituent')
plt.tight_layout()
plt.savefig('Plots/Correlations/Avg_correlations.pdf')
plt.close()

nanmean_corr_mtx =  np.nanmean(corr_mtx_per_const, axis = 0)

ax = sns.heatmap(nanmean_corr_mtx, linewidth=0.5, vmin = -1, vmax = 1, cmap = 'magma', yticklabels = feature_list, xticklabels = feature_list, square = True)
ax.set(title=f'Average correlations per constituent (ignoring nans)')
plt.tight_layout()
plt.savefig('Plots/Correlations/Avg_correlations_nonan.pdf')
plt.show()

print('Process complete!')