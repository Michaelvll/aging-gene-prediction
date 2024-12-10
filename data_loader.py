import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor


# import wandb
# wandb.init(project='gene')

#ct = 'DG_Glut'


DMR_FEATURE_NAMES = ['2mo', '9mo', '18mo', '9mo-2mo', '18mo-9mo', '18mo-2mo', 'log2(gene_length)', 'log2(r_length)', 'log2(r_length/gene_length)', 'log2(distance)']
CG_GENEBODY_FEATURE_NAMES = ['2mo', '9mo', '18mo', '9mo-2mo', '18mo-9mo', '18mo-2mo', 'log2(gene_length)','(9mo-2mo)*log2(gene_length)',  '(18mo-2mo)*log2(gene_length)', 
                            '(18mo-9mo)*log2(gene_length)','DMG','pvalue']
CH_GENEBODY_FEATURE_NAMES = ['2mo', '9mo', '18mo', '9mo-2mo', '18mo-9mo', '18mo-2mo', 'log2(gene_length)','(9mo-2mo)*log2(gene_length)',  '(18mo-2mo)*log2(gene_length)', 
                            '(18mo-9mo)*log2(gene_length)','DMG','pvalue']
ATAC_FEATURE_NAMES = ['2mo', '9mo', '18mo', 'log2(9mo/2mo)', 'log2(18mo/9mo)', 'log2(18mo/2mo)', 'log2(gene_length)', 'log2(distance)','DAR']
HIC_FEATURE_NAMES = [ 'Tanova', '2mo.Q', '9mo.Q', '18mo.Q','9mo-2mo.Q','18mo-9mo.Q', '18mo-2mo.Q',
                        'log2(gene_length)', 'log2(anchor1_distance)','log2(anchor2_distance)','Diff_Loop'] #'Qanova', 'Eanova',,'2mo.T', '9mo.T', '18mo.T','9mo-2mo.T', '18mo-9mo.T', '18mo-2mo.T', 
ABC_DMR_NAMES = ['2mo.activity', '2mo.contact', '2mo.abc_score', '9mo.activity','9mo.contact', '9mo.abc_score', 
                    '18mo.activity', '18mo.contact','18mo.abc_score', 'log2(eg_distance)','log2(gene_length)','log2(contact_distance)']
ABC_peak_NAMES = ['2mo.activity', '2mo.contact', '2mo.abc_score', '9mo.activity','9mo.contact', '9mo.abc_score', 
                    '18mo.activity', '18mo.contact','18mo.abc_score', 'log2(eg_distance)','log2(gene_length)','log2(contact_distance)']



DATA_FEATURE_NAMES = {
    'dmr': DMR_FEATURE_NAMES,
    'mcg_genebody': CG_GENEBODY_FEATURE_NAMES,
    'mch_genebody': CH_GENEBODY_FEATURE_NAMES,
    'atac': ATAC_FEATURE_NAMES,
    'hic_loop': HIC_FEATURE_NAMES,
    'hic_abc_dmr':ABC_DMR_NAMES,
    'hic_abc_peak':ABC_peak_NAMES 
}

DATA_FEATURE_NAMES_LIST = list(DATA_FEATURE_NAMES.keys())



def get_dmr_feat(ct):
    dmr = pd.read_csv(f'ml_input/{ct}/{ct}.aDMR_gene.csv')
    dmr_feat = dmr
    dmr_feat.rename(columns={'gene_name': 'gene'}, inplace=True)
    dmr_feat['9mo-2mo'] = dmr_feat['9mo'] - dmr_feat['2mo']
    dmr_feat['18mo-9mo'] = dmr_feat['18mo'] - dmr_feat['9mo']
    dmr_feat['18mo-2mo'] = dmr_feat['18mo'] - dmr_feat['2mo']
    dmr_feat['log2(gene_length)'] = np.log2((dmr_feat['gene_end'] - dmr_feat['gene_start']).abs().astype(np.float64))
    dmr_feat['log2(r_length)'] = np.log2((dmr_feat['end'] - dmr_feat['start']).abs().astype(np.float64) + 5)
    dmr_feat['log2(r_length/gene_length)'] = np.log2((dmr_feat['end'] - dmr_feat['start'])/(dmr_feat['gene_end'] - dmr_feat['gene_start'])+5)
    dmr_feat['log2(distance)'] = np.log2((dmr_feat['gene_start'] - dmr_feat['start']).abs().astype(np.float64)+ 5)
    dmr_feat = dmr_feat[['gene', *DMR_FEATURE_NAMES]]
    # drop if inf
    # dmr_feat= dmr_feat.replace([np.inf, -np.inf], np.nan)
    # dmr_feat= dmr_feat.dropna()

    assert dmr_feat.isna().sum().sum() == 0
    assert dmr_feat.isin([np.inf, -np.inf]).sum().sum() == 0
    print('Processed dmr data')
    return dmr_feat

def get_genebody_feat(ct):
    mcg_genebody = pd.read_csv(f'ml_input/{ct}/{ct}.mCG_genebody_gene.csv')
    mcg_genebody_feat = mcg_genebody
    mcg_genebody_feat.rename(columns={'gene_name': 'gene'}, inplace=True)
    mcg_genebody_feat['9mo-2mo'] = mcg_genebody_feat['9mo'] - mcg_genebody_feat['2mo']
    mcg_genebody_feat['18mo-9mo'] = mcg_genebody_feat['18mo'] - mcg_genebody_feat['9mo']
    mcg_genebody_feat['18mo-2mo'] = mcg_genebody_feat['18mo'] - mcg_genebody_feat['2mo']
    mcg_genebody_feat['log2(gene_length)'] = np.log2(mcg_genebody_feat['gene_length'])
    mcg_genebody_feat['(9mo-2mo)*log2(gene_length)'] = mcg_genebody_feat['9mo-2mo'] * mcg_genebody_feat['log2(gene_length)']
    mcg_genebody_feat['(18mo-9mo)*log2(gene_length)'] = mcg_genebody_feat['18mo-9mo'] * mcg_genebody_feat['log2(gene_length)']
    mcg_genebody_feat['(18mo-2mo)*log2(gene_length)'] = mcg_genebody_feat['18mo-2mo'] * mcg_genebody_feat['log2(gene_length)']
    mcg_genebody_feat = mcg_genebody_feat[['gene', *CG_GENEBODY_FEATURE_NAMES]]
    mcg_genebody_feat= mcg_genebody_feat.dropna()
    assert mcg_genebody_feat.isna().sum().sum() == 0
    assert mcg_genebody_feat.isin([np.inf, -np.inf]).sum().sum() == 0
    print('Processed mCG genebody data')
    return mcg_genebody_feat


def get_mch_genebody_feat(ct):
    mch_genebody = pd.read_csv(f'ml_input/{ct}/{ct}.mCH_genebody_gene.csv')
    mch_genebody_feat = mch_genebody
    mch_genebody_feat.rename(columns={'gene_name': 'gene'}, inplace=True)
    mch_genebody_feat['9mo-2mo'] = mch_genebody_feat['9mo'] - mch_genebody_feat['2mo']
    mch_genebody_feat['18mo-9mo'] = mch_genebody_feat['18mo'] - mch_genebody_feat['9mo']
    mch_genebody_feat['18mo-2mo'] = mch_genebody_feat['18mo'] - mch_genebody_feat['2mo']
    mch_genebody_feat['log2(gene_length)'] = np.log2(mch_genebody_feat['gene_length'])
    mch_genebody_feat['(9mo-2mo)*log2(gene_length)'] = mch_genebody_feat['9mo-2mo'] * mch_genebody_feat['log2(gene_length)']
    mch_genebody_feat['(18mo-9mo)*log2(gene_length)'] = mch_genebody_feat['18mo-9mo'] * mch_genebody_feat['log2(gene_length)']
    mch_genebody_feat['(18mo-2mo)*log2(gene_length)'] = mch_genebody_feat['18mo-2mo'] * mch_genebody_feat['log2(gene_length)']
    mch_genebody_feat = mch_genebody_feat[['gene', *CH_GENEBODY_FEATURE_NAMES]]
    mch_genebody_feat= mch_genebody_feat.dropna()
    assert mch_genebody_feat.isna().sum().sum() == 0
    assert mch_genebody_feat.isin([np.inf, -np.inf]).sum().sum() == 0
    print('Processed mCH genebody data')
    return mch_genebody_feat


def get_atac_feat(ct):
    atac = pd.read_csv(f'ml_input/{ct}/{ct}.peak_gene.csv')
    atac_feat = atac
    atac_feat.rename(columns={'gene_name': 'gene'}, inplace=True)
    atac_feat['log2(9mo/2mo)'] = np.log2(atac_feat['9mo'] + 1e-10) - np.log2(atac_feat['2mo'] + 1e-10)
    atac_feat['log2(18mo/9mo)'] = np.log2(atac_feat['18mo'] + 1e-10) - np.log2(atac_feat['9mo'] + 1e-10)
    atac_feat['log2(18mo/2mo)'] = np.log2(atac_feat['18mo'] + 1e-10) - np.log2(atac_feat['2mo'] + 1e-10)
    atac_feat['log2(gene_length)'] = np.log2((atac_feat['gene_end'] - atac_feat['gene_start']).abs().astype(np.float64) + 1e-10)
    atac_feat['log2(distance)'] = np.log2(atac_feat['distance'] + 1e-10)
    atac_feat = atac_feat[['gene', *ATAC_FEATURE_NAMES]]
    #check if any na or inf 
    assert atac_feat.isna().sum().sum() == 0
    assert atac_feat.isin([np.inf, -np.inf]).sum().sum() == 0
    print('Processed atac data')
    return atac_feat

def get_hic_feat(ct):
    hic = pd.read_csv(f'ml_input/{ct}/{ct}.Loop_gene.csv.gz')
    hic_feat = hic
    hic_feat.rename(columns={'gene_name': 'gene'}, inplace=True)
    hic_feat['9mo-2mo.Q'] = hic_feat['9mo.Q'] - hic_feat['2mo.Q']
    hic_feat['18mo-9mo.Q'] = hic_feat['18mo.Q'] - hic_feat['9mo.Q']
    hic_feat['18mo-2mo.Q'] = hic_feat['18mo.Q'] - hic_feat['2mo.Q']
    hic_feat['9mo-2mo.T'] = hic_feat['9mo.T'] - hic_feat['2mo.T']
    hic_feat['18mo-9mo.T'] = hic_feat['18mo.T'] - hic_feat['9mo.T']
    hic_feat['18mo-2mo.T'] = hic_feat['18mo.T'] - hic_feat['2mo.T']
    hic_feat['log2(gene_length)'] = np.log2(hic_feat['gene_length'] )
    hic_feat['log2(anchor1_distance)'] = np.log2(hic_feat['anchor1_distance'] + 10000) #10000 i the loop resolution
    hic_feat['log2(anchor2_distance)'] = np.log2(hic_feat['anchor2_distance'] + 10000)

    hic_feat = hic_feat[['gene', *HIC_FEATURE_NAMES]]
    assert hic_feat.isna().sum().sum() == 0
    assert hic_feat.isin([np.inf, -np.inf]).sum().sum() == 0
    print('Processed hic loop data')
    return hic_feat


def get_abc_dmr_feat(ct):
    abc_dmr = pd.read_csv(f'ml_input/{ct}/{ct}.abc_enhancer.DMR_gene.csv').fillna(0)
    abc_dmr_feat = abc_dmr
    abc_dmr_feat.rename(columns={'gene_name': 'gene'}, inplace=True)
    abc_dmr_feat['log2(eg_distance)'] = np.log2(np.minimum(abs(abc_dmr_feat['start'] - abc_dmr_feat['gene_start']), abs(abc_dmr_feat['start'] - abc_dmr_feat['gene_end'])) + 10000)
    abc_dmr_feat['log2(gene_length)'] = np.log2(abc_dmr_feat['gene_end'] - abc_dmr_feat['gene_start'])
    abc_dmr_feat['log2(contact_distance)'] = np.log2(abs(abc_dmr_feat['end'] - abc_dmr_feat['start']) + 10000)
    abc_dmr_feat = abc_dmr_feat[['gene', *ABC_DMR_NAMES]]
    assert abc_dmr_feat.isna().sum().sum() == 0
    assert abc_dmr_feat.isin([np.inf, -np.inf]).sum().sum() == 0
    print('Processed abc dmr data')
    return abc_dmr_feat



def get_abc_peak_feat(ct):
    abc_peak =  pd.read_csv(f'ml_input/{ct}/{ct}.abc_enhancer.peak_gene.csv').fillna(0)
    abc_peak_feat = abc_peak
    abc_peak_feat.rename(columns={'gene_name': 'gene'}, inplace=True)
    abc_peak_feat['log2(eg_distance)'] = np.log2(np.minimum(abs(abc_peak_feat['start'] - abc_peak_feat['gene_start']), abs(abc_peak_feat['start'] - abc_peak_feat['gene_end'])) + 10000)
    abc_peak_feat['log2(gene_length)'] = np.log2(abc_peak_feat['gene_end'] - abc_peak_feat['gene_start'])
    abc_peak_feat['log2(contact_distance)'] = np.log2(abs(abc_peak_feat['end'] - abc_peak_feat['start']) + 10000)
    abc_peak_feat = abc_peak_feat[['gene', *ABC_peak_NAMES]]
    assert abc_peak_feat.isna().sum().sum() == 0
    assert abc_peak_feat.isin([np.inf, -np.inf]).sum().sum() == 0
    print('Processed abc peak data')
    return abc_peak_feat

def load_data(ct):
    df = pd.read_csv(f'ml_input/{ct}/{ct}.luisa_RNA_DEG.csv', index_col =0)
    #df.set_index('gene', inplace=True)

    # non_zero_genes = df[df['DEG'] != 0].index
    # df = df[df.index.isin(non_zero_genes)]
    #gene2value = df[['-log10(fdr)','log2(old/young)', 'DEG']]
    gene2value = df[['DEG']]

    # df = df[df.index.isin(non_zero_genes)]
    # use all pleak/loop, adding columns as pvalue, anova, 
 
    DATA = {}

    FEATURE_LOADING_FUNCTIONS = {
        'dmr': get_dmr_feat,
        'mcg_genebody': get_genebody_feat,
        'mch_genebody': get_mch_genebody_feat,
        'atac': get_atac_feat,
        'hic_loop': get_hic_feat,
        'hic_abc_dmr': get_abc_dmr_feat,
        'hic_abc_peak': get_abc_peak_feat
    }
    
    # Define tasks to run in parallel
    tasks = [
        (name, FEATURE_LOADING_FUNCTIONS[name]) for name in DATA_FEATURE_NAMES_LIST
    ]

    # Run tasks in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor() as executor:
        # Submit all tasks and store futures
        futures = {
            name: executor.submit(func, ct) 
            for name, func in tasks
        }
        
        # Get results as they complete and store in DATA
        for name, future in futures.items():
            DATA[name] = future.result()

    index_order = gene2value.index.tolist()
    # Train a sequence model on dmr_feat to predict gene2value['log2(old/young)']
    # Each gene has a sequence of 4 features, 2mo, 9mo, 18mo, old-young
    # The sequence length is not fixed, so we need to use a dynamic model
    # Let's use a commonly used sequence prediction model for sentence classification
    # like LSTM or Transformer

    X = {}
    # Step 1: Prepare the data
    print('Preparing data')
    def prepare_data(feature_type):
        features = DATA[feature_type]
        feature_names = DATA_FEATURE_NAMES[feature_type]
        list_feat = features.groupby('gene').apply(lambda x: x[feature_names].values.tolist())
        list_feat = list_feat.reindex(index_order, fill_value=[[0] * len(feature_names)])
        return list_feat.values.tolist()
    with ThreadPoolExecutor() as executor:
        # Submit all tasks and store futures
        futures = {
            feature_type: executor.submit(prepare_data, feature_type) 
            for feature_type in DATA
        }
        
        # Get results as they complete and store in DATA
        for feature_type, future in futures.items():
            X[feature_type] = future.result()

    # y = gene2value['DEG'].values.tolist()
    # y = np.array([int(i) for i in y])

    y = gene2value.values

    return {
        'y': y,
        'X': X,
    }

def get_balanced_data_by_samllest_group(data):
    # Separate the data into zero and non-zero y values
    y = data['y']
    # zero_indices = np.where(y[:, 2] == 0)[0]
    # non_zero_indices = np.where(y[:, 2] != 0)[0]
    zero_indices = np.where(y == 0)[0]
    up_indices = np.where(y == 1)[0]
    down_indices = np.where(y == -1)[0]
    print(f'zero: {len(zero_indices)}, up: {len(up_indices)},down: {len(down_indices)}')

    # get the samllest group and sample the same amount from other two
    min_len = min(len(zero_indices), len(up_indices), len(down_indices))
    zero_indices = np.random.choice(zero_indices, min_len, replace=False)
    up_indices = np.random.choice(up_indices, min_len, replace=False)
    down_indices = np.random.choice(down_indices, min_len, replace=False)
    # concat
    sampled_indices = np.concatenate([zero_indices, up_indices, down_indices])

    # Create balanced dataset
    X_balanced = {}

    print('Getting balanced data')
    def index_features(feature_type):
        features = data['X'][feature_type]
        return [features[i] for i in sampled_indices]

    with ThreadPoolExecutor() as executor:
        # Submit all tasks and store futures
        futures = {
            feature_type: executor.submit(index_features, feature_type) 
            for feature_type in data['X']
        }
        # Get results as they complete and store in X_balanced
        for feature_type, future in futures.items():
            X_balanced[feature_type] = future.result()
    y_balanced = data['y'][sampled_indices, :]
    return X_balanced, y_balanced

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

    print('Getting balanced data')
    def index_features(feature_type):
        features = data['X'][feature_type]
        return [features[i] for i in sampled_indices]

    with ThreadPoolExecutor() as executor:
        # Submit all tasks and store futures
        futures = {
            feature_type: executor.submit(index_features, feature_type) 
            for feature_type in data['X']
        }
        # Get results as they complete and store in X_balanced
        for feature_type, future in futures.items():
            X_balanced[feature_type] = future.result()
    y_balanced = data['y'][sampled_indices, :]
    return X_balanced, y_balanced

def concat_group_data(concat_cts):
    y_balanced = []
    X_balanced_list = []

    for ct in concat_cts:
        print()
        data_tmp = load_data(ct=ct)
        X_balanced_tmp, y_balanced_tmp = get_balanced_data_by_samllest_group(data_tmp)
        y_balanced.append(y_balanced_tmp)
        X_balanced_list.append(X_balanced_tmp)

    y_balanced = np.concatenate(y_balanced)
    X_balanced = defaultdict(list)

    for feature_name in DATA_FEATURE_NAMES_LIST:
        for i in range(len(concat_cts)):
            X_balanced[feature_name] += X_balanced_list[i][feature_name]

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


