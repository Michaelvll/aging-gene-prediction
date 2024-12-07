import pandas as pd
import numpy as np

# import wandb
# wandb.init(project='gene')

ct = 'DG_Glut'

def load_data():
    df = pd.read_csv(f'data/{ct}/{ct}.luisa_RNA_DEG.csv', index_col =0)
    #df.set_index('gene', inplace=True)

    # non_zero_genes = df[df['DEG'] != 0].index

    # df = df[df.index.isin(non_zero_genes)]
    gene2value = df[['DEG']]
    #gene2value = df[['-log10(fdr)','log2(old/young)', 'DEG']]

    # df = df[df.index.isin(non_zero_genes)]
# use all pleak/loop, adding columns as pvalue, anova, 
    MCG_FEATURE_NAMES = ['2mo', '9mo', '18mo', '9mo-2mo', '18mo-9mo', '18mo-2mo', 'log2(gene_length)', 'log2(r_length)', 'log2(r_length/gene_length)', 'log2(distance)']
    GENEBODY_FEATURE_NAMES = ['2mo', '9mo', '18mo', '9mo-2mo', '18mo-9mo', '18mo-2mo', 'log2(gene_length)','(9mo-2mo)*log2(gene_length)', 
                            '(18mo-2mo)*log2(gene_length)', '(18mo-9mo)*log2(gene_length)','DMG','corrected_pvalue']
    ATAC_FEATURE_NAMES = ['2mo', '9mo', '18mo', 'log2(9mo/2mo)', 'log2(18mo/9mo)', 'log2(18mo/2mo)', 'log2(gene_length)', 'log2(r_length)', 'log2(r_length/gene_length)', 'log2(distance)','DAR']
    HIC_FEATURE_NAMES = [ 'Tanova', '2mo.Q', '9mo.Q', '18mo.Q','9mo-2mo.Q','18mo-9mo.Q', '18mo-2mo.Q',
                         'log2(gene_length)', 'log2(a_length)', 'log2(a_length/gene_length)','Diff_Loop'] #'Qanova', 'Eanova',,'2mo.T', '9mo.T', '18mo.T','9mo-2mo.T', '18mo-9mo.T', '18mo-2mo.T', 

    DATA_FEATURE_NAMES = {
        'mcg': MCG_FEATURE_NAMES,
        'atac': ATAC_FEATURE_NAMES,
        'hic': HIC_FEATURE_NAMES,
        'genebody': GENEBODY_FEATURE_NAMES
    }
    DATA = {}

    mcg = pd.read_csv(f'data/{ct}/{ct}.aDMR_gene.csv')
    mcg_feat = mcg
    mcg_feat.rename(columns={'gene_name': 'gene'}, inplace=True)
    mcg_feat['9mo-2mo'] = mcg_feat['9mo'] - mcg_feat['2mo']
    mcg_feat['18mo-9mo'] = mcg_feat['18mo'] - mcg_feat['9mo']
    mcg_feat['18mo-2mo'] = mcg_feat['18mo'] - mcg_feat['2mo']
    mcg_feat['log2(gene_length)'] = np.log2((mcg_feat['gene_end'] - mcg_feat['gene_start']).abs().astype(np.float64))
    mcg_feat['log2(r_length)'] = np.log2((mcg_feat['end'] - mcg_feat['start']).abs().astype(np.float64))
    mcg_feat['log2(r_length/gene_length)'] = np.log2((mcg_feat['end'] - mcg_feat['start'])/(mcg_feat['gene_end'] - mcg_feat['gene_start']))
    mcg_feat['log2(distance)'] = np.log2((mcg_feat['gene_start'] - mcg_feat['start']).abs().astype(np.float64))
    mcg_feat = mcg_feat[['gene', *MCG_FEATURE_NAMES]]

    assert mcg_feat.isna().sum().sum() == 0
    assert mcg_feat.isin([np.inf, -np.inf]).sum().sum() == 0

    DATA['mcg'] = mcg_feat
    print('Processed mcg data')

       
    genebody = pd.read_csv(f'data/{ct}/{ct}.mCG_genebody_gene.csv')
    genebody_feat = genebody
    genebody_feat.rename(columns={'gene_name': 'gene'}, inplace=True)
    genebody_feat['9mo-2mo'] = genebody_feat['9mo'] - genebody_feat['2mo']
    genebody_feat['18mo-9mo'] = genebody_feat['18mo'] - genebody_feat['9mo']
    genebody_feat['18mo-2mo'] = genebody_feat['18mo'] - genebody_feat['2mo']

    genebody_feat['log2(gene_length)'] = np.log2(genebody_feat['gene_length'])

    genebody_feat['(9mo-2mo)*log2(gene_length)'] = genebody_feat['9mo-2mo'] * genebody_feat['log2(gene_length)']
    genebody_feat['(18mo-9mo)*log2(gene_length)'] = genebody_feat['18mo-9mo'] * genebody_feat['log2(gene_length)']
    genebody_feat['(18mo-2mo)*log2(gene_length)'] = genebody_feat['18mo-2mo'] * genebody_feat['log2(gene_length)']

    genebody_feat = genebody_feat[['gene', *GENEBODY_FEATURE_NAMES]]
    genebody_feat= genebody_feat.dropna()
    
    assert genebody_feat.isna().sum().sum() == 0
    assert genebody_feat.isin([np.inf, -np.inf]).sum().sum() == 0

    DATA['genebody'] = genebody_feat
    print('Processed genebody data')
    
    
    atac = pd.read_csv(f'data/{ct}/{ct}.peak_gene.csv')
    atac_feat = atac
    atac_feat.rename(columns={'gene_name': 'gene'}, inplace=True)
    atac_feat['log2(9mo/2mo)'] = np.log2(atac_feat['9mo'] + 1e-10) - np.log2(atac_feat['2mo'] + 1e-10)
    atac_feat['log2(18mo/9mo)'] = np.log2(atac_feat['18mo'] + 1e-10) - np.log2(atac_feat['9mo'] + 1e-10)
    atac_feat['log2(18mo/2mo)'] = np.log2(atac_feat['18mo'] + 1e-10) - np.log2(atac_feat['2mo'] + 1e-10)
    atac_feat['log2(gene_length)'] = np.log2((atac_feat['gene_end'] - atac_feat['gene_start']).abs().astype(np.float64) + 1e-10)
    atac_feat['log2(r_length)'] = np.log2((atac_feat['peak_end'] - atac_feat['peak_start']).abs().astype(np.float64) + 1e-10)
    atac_feat['log2(r_length/gene_length)'] = atac_feat['log2(r_length)'] - atac_feat['log2(gene_length)']
    atac_feat['log2(distance)'] = np.log2((atac_feat['gene_start'] - atac_feat['peak_start']).abs().astype(np.float64) + 1e-10)
    atac_feat = atac_feat[['gene', *ATAC_FEATURE_NAMES]]
    #check if any na or inf 
    assert atac_feat.isna().sum().sum() == 0
    assert atac_feat.isin([np.inf, -np.inf]).sum().sum() == 0
    DATA['atac'] = atac_feat
    print('Processed atac data')

    
    hic = pd.read_csv(f'data/{ct}/{ct}.Loop_gene.csv.gz')
    hic.columns = ['chrom','anchor1_start','anchor1_end','chrom','anchor2_start','anchor2_end',
                'Qanova','Eanova','Tanova','2mo.Q','9mo.Q','18mo.Q','2mo.T','9mo.T','18mo.T','Diff_Loop',
                'gene_chrom','gene_start','gene_end','gene_id','strand','gene_name','gene_type']
    #hic = pd.read_csv('data/Oligo_NN.diff_loop_gene.csv')
    hic_feat = hic
    hic_feat.rename(columns={'gene_name': 'gene'}, inplace=True)
    hic_feat['9mo-2mo.Q'] = hic_feat['9mo.Q'] - hic_feat['2mo.Q']
    hic_feat['18mo-9mo.Q'] = hic_feat['18mo.Q'] - hic_feat['9mo.Q']
    hic_feat['18mo-2mo.Q'] = hic_feat['18mo.Q'] - hic_feat['2mo.Q']

    hic_feat['9mo-2mo.T'] = hic_feat['9mo.T'] - hic_feat['2mo.T']
    hic_feat['18mo-9mo.T'] = hic_feat['18mo.T'] - hic_feat['9mo.T']
    hic_feat['18mo-2mo.T'] = hic_feat['18mo.T'] - hic_feat['2mo.T']

    hic_feat['log2(gene_length)'] = np.log2((hic_feat['gene_end'] - hic_feat['gene_start']).abs().astype(np.float64) + 1e-10)
    hic_feat['log2(a_length)'] = np.log2((hic_feat['anchor2_start'] - hic_feat['anchor1_start']).abs().astype(np.float64) + 10000) #10000 i the loop resolution
    hic_feat['log2(a_length/gene_length)'] = hic_feat['log2(a_length)'] - hic_feat['log2(gene_length)']
    hic_feat = hic_feat[['gene', *HIC_FEATURE_NAMES]]
    assert hic_feat.isna().sum().sum() == 0
    assert hic_feat.isin([np.inf, -np.inf]).sum().sum() == 0

    DATA['hic'] = hic_feat
    print('Processed hic data')

    index_order = gene2value.index.tolist()
    # Train a sequence model on mcg_feat to predict gene2value['log2(old/young)']
    # Each gene has a sequence of 4 features, 2mo, 9mo, 18mo, old-young
    # The sequence length is not fixed, so we need to use a dynamic model
    # Let's use a commonly used sequence prediction model for sentence classification
    # like LSTM or Transformer

    X = {}
    # Step 1: Prepare the data
    for feature_type, features in DATA.items():
        feature_names = DATA_FEATURE_NAMES[feature_type]
        list_feat = features.groupby('gene').apply(lambda x: x[feature_names].values.tolist())
        list_feat = list_feat.reindex(index_order, fill_value=[[0] * len(feature_names)])
        X[feature_type] = list_feat.values.tolist()

    # y = gene2value['DEG'].values.tolist()
    # y = np.array([int(i) for i in y])

    y = gene2value.values

    return {
        'y': y,
        'X': X,
    }

def get_balanced_data(data):
    # Separate the data into zero and non-zero y values
    y = data['y']
    # zero_indices = np.where(y[:, 2] == 0)[0]
    # non_zero_indices = np.where(y[:, 2] != 0)[0]
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
    for feature_type, features in data['X'].items():
        X_balanced[feature_type] = [features[i] for i in sampled_indices]
    y_balanced = data['y'][sampled_indices, :]
    return X_balanced, y_balanced


# Normalization function
def normalize_features(train_data, test_data):
    # Flatten the lists for easier processing
    # train data: [N, L, F] -> [N*L, F]
    train_flat = [item for sublist in train_data for item in sublist]
    test_flat = [item for sublist in test_data for item in sublist]
    
    # Convert to numpy arrays
    train_array = np.array(train_flat)
    test_array = np.array(test_flat)
    
    # Normalize all features using min-max scaling based on train data
    min_vals = np.min(train_array, axis=0)
    max_vals = np.max(train_array, axis=0)
    train_normalized = (train_array - min_vals) / (max_vals - min_vals)
    test_normalized = (test_array - min_vals) / (max_vals - min_vals)
    
    assert np.all(max_vals - min_vals != 0), np.where(max_vals - min_vals == 0)
    
    # Reconstruct the data structure
    def reconstruct_data(normalized_array, original_data):
        normalized_data = []
        idx = 0
        for sublist in original_data:
            normalized_sublist = []
            for _ in sublist:
                normalized_sublist.append(normalized_array[idx].tolist())
                idx += 1
            normalized_data.append(normalized_sublist)
        return normalized_data
    
    train_normalized = reconstruct_data(train_normalized, train_data)
    test_normalized = reconstruct_data(test_normalized, test_data)
    
    return train_normalized, test_normalized


