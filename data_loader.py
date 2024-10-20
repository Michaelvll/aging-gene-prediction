import pandas as pd
import numpy as np

import wandb
wandb.init(project='gene')

def load_data():
    data = {}
    df = pd.read_csv('data/Oligo_NN.RNA_DEG.csv')
    df.set_index('gene', inplace=True)

    # non_zero_genes = df[df['DEG'] != 0].index

    # df = df[df.index.isin(non_zero_genes)]
    gene2value = df[['DEG']]

    MCG_FEATURE_NAMES = ['2mo', '9mo', '18mo', '9mo-2mo', '18mo-9mo', '9mo/2mo', '18mo/9mo', 'old-young', 'old/young', 'gene_length', 'distance']
    ATAC_FEATURE_NAMES = ['log2(old/young)', 'gene_length', 'distance']

    # Process mcg data
    mcg = pd.read_csv('data/Oligo_NN.aDMR_gene.csv')
    mcg_feat = mcg
    mcg_feat.rename(columns={'gene_name': 'gene'}, inplace=True)
    mcg_feat['distance'] = (mcg_feat['gene_start'] - mcg_feat['start']).abs().astype(np.float64)
    mcg_feat['old/young'] = mcg_feat['18mo'] / mcg_feat['2mo']
    mcg_feat['9mo-2mo'] = mcg_feat['9mo'] - mcg_feat['2mo']
    mcg_feat['18mo-9mo'] = mcg_feat['18mo'] - mcg_feat['9mo']
    mcg_feat['9mo/2mo'] = mcg_feat['9mo'] / mcg_feat['2mo']
    mcg_feat['18mo/9mo'] = mcg_feat['18mo'] / mcg_feat['9mo']
    mcg_feat['gene_length'] = mcg_feat['gene_end'] - mcg_feat['gene_start']
    mcg_feat.fillna(0, inplace=True)
    mcg_feat = mcg_feat[['gene', *MCG_FEATURE_NAMES]]


    atac = pd.read_csv('data/Oligo_NN.ATAC_gene.csv')
    atac_feat = atac
    atac_feat.rename(columns={'gene_name': 'gene'}, inplace=True)
    atac_feat['distance'] = (atac_feat['gene_start'] - atac_feat['start']).abs().astype(np.float64)
    atac_feat['gene_length'] = atac_feat['gene_end'] - atac_feat['gene_start']
    atac_feat = atac_feat[['gene', *ATAC_FEATURE_NAMES]]

    # print('mcg corr:', mcg_mean.corrwith(gene2value['DEG']))
    # print('atac corr:', atac_mean.corrwith(gene2value['DEG']))
    index_order = gene2value.index.tolist()


    # Train a sequence model on mcg_feat to predict gene2value['log2(old/young)']
    # Each gene has a sequence of 4 features, 2mo, 9mo, 18mo, old-young
    # The sequence length is not fixed, so we need to use a dynamic model
    # Let's use a commonly used sequence prediction model for sentence classification
    # like LSTM or Transformer

    # Step 1: Prepare the data
    list_mcg_feat = mcg_feat.groupby('gene').apply(lambda x: x[MCG_FEATURE_NAMES].values.tolist())
    list_mcg_feat = list_mcg_feat.reindex(index_order, fill_value=[[0] * len(MCG_FEATURE_NAMES)])
    x_mcg = list_mcg_feat.values.tolist()

    list_atac_feat = atac_feat.groupby('gene').apply(lambda x: x[ATAC_FEATURE_NAMES].values.tolist())
    list_atac_feat = list_atac_feat.reindex(index_order, fill_value=[[0] * len(ATAC_FEATURE_NAMES)])
    x_atac = list_atac_feat.values.tolist()

    y = gene2value.loc[list_mcg_feat.index]['DEG'].values.tolist()
    y = np.array([int(i) for i in y])
    y[:10]

    data['mcg'] = x_mcg
    data['atac'] = x_atac
    data['y'] = y
    return data


def get_balanced_data(data):
    # Separate the data into zero and non-zero y values
    y = data['y']
    zero_indices = np.where(y == 0)[0]
    non_zero_indices = np.where(y != 0)[0]
    print(f'zero: {len(zero_indices)}, non-zero: {len(non_zero_indices)}')

    # Sample len(non_zero_indices) indices from each group
    n_samples = len(non_zero_indices)
    sampled_zero_indices = np.random.choice(zero_indices, n_samples // 2, replace=False)
    sampled_non_zero_indices = np.random.choice(non_zero_indices, n_samples, replace=False)

    # Combine the sampled indices
    sampled_indices = np.concatenate([sampled_zero_indices, sampled_non_zero_indices])

    # Create balanced dataset
    X_balanced = {}
    X_balanced['mcg'] = [data['mcg'][i] for i in sampled_indices]
    X_balanced['atac'] = [data['atac'][i] for i in sampled_indices]
    y_balanced = data['y'][sampled_indices]
    return X_balanced, y_balanced


# Normalization function
def normalize_features(train_data, test_data):
    # Flatten the lists for easier processing
    train_flat = [item for sublist in train_data for item in sublist]
    test_flat = [item for sublist in test_data for item in sublist]
    feature_dim = len(train_flat[0])
    # Separate features
    train_other_features = np.array([item[:feature_dim-1] for item in train_flat])
    train_distances = np.array([item[feature_dim-1] for item in train_flat])
    test_other_features = np.array([item[:feature_dim-1] for item in test_flat])
    test_distances = np.array([item[feature_dim-1] for item in test_flat])
    
    # Normalize other features using min-max scaling based on train data
    min_vals = np.min(train_other_features, axis=0)
    max_vals = np.max(train_other_features, axis=0)
    train_normalized_features = (train_other_features - min_vals) / (max_vals - min_vals)
    test_normalized_features = (test_other_features - min_vals) / (max_vals - min_vals)
    
    # Normalize distances using log transformation and then min-max scaling based on train data
    train_log_distances = np.log1p(train_distances)
    test_log_distances = np.log1p(test_distances)
    min_dist = np.min(train_log_distances)
    max_dist = np.max(train_log_distances)
    train_normalized_distances = (train_log_distances - min_dist) / (max_dist - min_dist)
    test_normalized_distances = (test_log_distances - min_dist) / (max_dist - min_dist)
    
    
    # Combine normalized features and distances
    def reconstruct_data(features, distances, original_data):
        normalized_data = []
        idx = 0
        for sublist in original_data:
            normalized_sublist = []
            for _ in sublist:
                normalized_sublist.append(list(features[idx][:feature_dim-1]) + [distances[idx]])
                idx += 1
            normalized_data.append(normalized_sublist)
        return normalized_data
    
    train_normalized = reconstruct_data(train_normalized_features, train_normalized_distances, train_data)
    test_normalized = reconstruct_data(test_normalized_features, test_normalized_distances, test_data)
    
    return train_normalized, test_normalized


