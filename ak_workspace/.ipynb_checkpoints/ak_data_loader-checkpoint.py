import pandas as pd
import numpy as np

# import wandb
# wandb.init(project='gene')



# Setting the default loaded features (This can be changed once the calculated features are added
ENHANCER_DMR_FEATURES = ['9mo.2mo.activity_diff.mean', '18mo.2mo.activity_diff.mean', '18mo.9mo.activity_diff.mean', '9mo.2mo.contact_diff.mean',
                         '18mo.2mo.contact_diff.mean', '18mo.9mo.contact_diff.mean', '9mo.2mo.abc_score_diff.mean', '18mo.2mo.abc_score_diff.mean',
                         '18mo.9mo.abc_score_diff.mean', '9mo.2mo.activity_diff.max', '18mo.2mo.activity_diff.max', '18mo.9mo.activity_diff.max',
                         '9mo.2mo.contact_diff.max', '18mo.2mo.contact_diff.max', '18mo.9mo.contact_diff.max', '9mo.2mo.abc_score_diff.max',
                         '18mo.2mo.abc_score_diff.max', '18mo.9mo.abc_score_diff.max', '9mo.2mo.activity_diff.min', '18mo.2mo.activity_diff.min',
                         '18mo.9mo.activity_diff.min', '9mo.2mo.contact_diff.min', '18mo.2mo.contact_diff.min', '18mo.9mo.contact_diff.min',
                         '9mo.2mo.abc_score_diff.min', '18mo.2mo.abc_score_diff.min', '18mo.9mo.abc_score_diff.min']

ENHANCER_PEAK_FEATURES = ['9mo.2mo.activity_diff.mean', '18mo.2mo.activity_diff.mean', '18mo.9mo.activity_diff.mean', '9mo.2mo.contact_diff.mean',
                         '18mo.2mo.contact_diff.mean', '18mo.9mo.contact_diff.mean', '9mo.2mo.abc_score_diff.mean', '18mo.2mo.abc_score_diff.mean',
                         '18mo.9mo.abc_score_diff.mean', '9mo.2mo.activity_diff.max', '18mo.2mo.activity_diff.max', '18mo.9mo.activity_diff.max',
                         '9mo.2mo.contact_diff.max', '18mo.2mo.contact_diff.max', '18mo.9mo.contact_diff.max', '9mo.2mo.abc_score_diff.max',
                         '18mo.2mo.abc_score_diff.max', '18mo.9mo.abc_score_diff.max', '9mo.2mo.activity_diff.min', '18mo.2mo.activity_diff.min',
                         '18mo.9mo.activity_diff.min', '9mo.2mo.contact_diff.min', '18mo.2mo.contact_diff.min', '18mo.9mo.contact_diff.min',
                         '9mo.2mo.abc_score_diff.min', '18mo.2mo.abc_score_diff.min', '18mo.9mo.abc_score_diff.min']


DEFAULT_DATA_FEATURE_NAMES = {'enhancer_DMR' : ENHANCER_DMR_FEATURES,
                             'enhancer_peak' : ENHANCER_PEAK_FEATURES,
                              # 'DAR' : 'aDAR_gene.csv', 
                              # 'DMR' : 'aDMR_gene.csv',
                              # 'loops' : 'Loop_gene.csv.gz', 
                              # 'mcg_genebody' : 'mCG_genebody_gene.csv', 
                              # 'mch_genebody' : 'mCH_genebody_gene.csv', 
                              # 'atac' : 'peak_gene.csv'
                             }

def load_data(y_val = "DEG", rna_type='luisa', data_filepath="data", ct="Oligo_NN", DATA_FEATURE_NAMES=DEFAULT_DATA_FEATURE_NAMES, na_cutoff = 0.5):
    """
    author: amit klein / rachel zeng
    email: a3klein@ucsd.edu

    y_val: DEG / logFC (for classification or regression), 
    # rna_filepath: The filepath to the csv with RNA DEGs
    rna_type: "luisa" / "in_house" 
    data_filepath: filepath to the engineered feature csv files
    ct: The Cell Type to return data for
    DATA_FEATURE_NAMES: 
        a dictionary with the keys as the data modalities to return and the values as the names of the features to return for that modality. 
    na_cutoff  (int) : the ratio of missing values within a given feature to remove that feature from the resturned dataframe
    """
    # df = pd.read_csv(f'{data_filepath}/{ct}/{ct}.rna.csv', index_col=0)
    if rna_type == "luisa": 
        df = pd.read_csv(f'{data_filepath}/{ct}/{ct}.luisa_rna.csv', index_col=0)
    elif rna_type == "in_house": 
        df = pd.read_csv(f'{data_filepath}/{ct}/{ct}.inhouse_rna.csv', index_col=0)
    else: 
        raise(f"Invalid rna_type name {rna_type}")
    
    # df.set_index('gene', inplace=True)

    if y_val == "DEG": 
        gene2value = df[['DEG']]
    elif y_val == "logFC": 
        gene2value = df[['avg_log2FC']]

    DATA = {}

    for _feat, FEATURES in DATA_FEATURE_NAMES.items(): 
        skip_bool = True
        try: # Trying to read the filepath for the data
            df_feat = pd.read_csv(f'{data_filepath}/{ct}/{ct}.{_feat}.csv')
        except: 
            skip_bool = False
            print(f"No {_feat} data filepath for this cell type: {ct}, Skipping this data modality")
            print(f"Missing Filepath: {data_filepath}/{ct}/{ct}.{_feat}.csv ")
        if skip_bool: # Calculating all features and the subsetting only for the ones to keep (Can be improved?)
            df_feat = df_feat[['gene_name', *FEATURES]]
            
            # Filtering out columns by nan ratio
            cols_to_keep = df_feat.columns[df_feat.isna().sum() / df_feat.shape[0] < na_cutoff]
            df_feat = df_feat[cols_to_keep].copy()
            
            # TODO: Fix this fillna thing when making the feature dataframes: 
            df_feat = df_feat.fillna(0)
            
            # Checking that the features are valid
            assert df_feat.isna().sum().sum() == 0
            assert df_feat.isin([np.inf, -np.inf]).sum().sum() == 0
    
            # Adding MCG features to the data dictionary
            DATA[_feat] = df_feat
            print(f'Processed {_feat} data')

    index_order = gene2value.index.tolist()

    X = {}
    # Step 1: Prepare the data
    for feature_type, features in DATA.items():
        feature_names = DATA_FEATURE_NAMES[feature_type]
        # list_feat = features.groupby('gene').apply(lambda x: x[feature_names].values.tolist(), include_groups=False)
        features = features.set_index('gene_name').reindex(index_order, fill_value = 0)
        X[feature_type] = features


    # handling Y
    genes = gene2value.index.values.tolist()
    if y_val == "DEG": 
        y = gene2value['DEG'].values.tolist()
        y = np.array([int(i) for i in y])
        Y = gene2value['DEG'].astype(int)
    elif y_val == "logFC": 
        y = gene2value['avg_log2FC'].values.tolist()
        y = np.array([float(i) for i in y])
        Y = gene2value['avg_log2FC'].astype(float)

    return {
        'y': Y,
        'X': X,
    }



def get_balanced_data(data, method=None, y_val='DEG'):
    # Separate the data into zero and non-zero y values
    y = data['y']
    if y_val=='DEG': 
        if method == 'balanced': 
            zero_indices = np.where(y == 0)[0]
            down_indices = np.where(y == -1)[0]
            up_indices = np.where(y == 1)[0]
            print(f'zero: {len(zero_indices)}, down: {len(down_indices)}, up: {len(up_indices)}')
            # Sample len(non_zero_indices) indices from each group
            n_samples = min(len(zero_indices), len(down_indices), len(up_indices))
            sampled_zero_indices = np.random.choice(zero_indices, n_samples, replace=False)
            sampled_down_indices = np.random.choice(down_indices, n_samples, replace=False)
            sampled_up_indices = np.random.choice(up_indices, n_samples, replace=False)
            # Combine the sampled indices
            sampled_indices = np.concatenate([sampled_zero_indices, sampled_down_indices, sampled_up_indices])
        else: 
            zero_indices = np.where(y == 0)[0]
            non_zero_indices = np.where(y != 0)[0]
            print(f'zero: {len(zero_indices)}, non-zero: {len(non_zero_indices)}')
        
            # Sample len(non_zero_indices) indices from each group
            n_samples = len(non_zero_indices)
            # sampled_zero_indices = np.random.choice(zero_indices, n_samples // 2, replace=False)
            sampled_zero_indices = np.random.choice(zero_indices, n_samples, replace=False)
            sampled_non_zero_indices = np.random.choice(non_zero_indices, n_samples, replace=False)
        
            # Combine the sampled indices
            sampled_indices = np.concatenate([sampled_zero_indices, sampled_non_zero_indices])
    elif y_val == 'logFC': 
        pass

    # Create balanced dataset
    X_balanced = {}
    for feature_type, features in data['X'].items():
        X_balanced[feature_type] = features.iloc[sampled_indices]
    y_balanced = data['y'].iloc[sampled_indices]
    return X_balanced, y_balanced


# Normalization function
def normalize_features(train_data, test_data):
    # Flatten the lists for easier processing
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


# Setting the default loaded features (This can be changed once the calculated features are added
DMR_FEATURE_NAMES = ['2mo', '9mo', '18mo', '9mo-2mo', '18mo-9mo', '18mo-2mo', 'log2(gene_length)', 'log2(r_length)', 'log2(r_length/gene_length)', 'log2(distance)']

MCG_GENEBODY_FEATURE_NAMES = ['2mo', '9mo', '18mo', '9mo-2mo', '18mo-9mo', '18mo-2mo', 'log2(gene_length)','(9mo-2mo)*log2(gene_length)', 
                        '(18mo-2mo)*log2(gene_length)', '(18mo-9mo)*log2(gene_length)','DMG','corrected_pvalue']
MCH_GENEBODY_FEATURE_NAMES = ['2mo', '9mo', '18mo', '9mo-2mo', '18mo-9mo', '18mo-2mo', 'log2(gene_length)','(9mo-2mo)*log2(gene_length)', 
                        '(18mo-2mo)*log2(gene_length)', '(18mo-9mo)*log2(gene_length)','DMG','corrected_pvalue']

ATAC_FEATURE_NAMES = ['2mo', '9mo', '18mo', 'log2(9mo/2mo)', 'log2(18mo/9mo)', 'log2(18mo/2mo)', 'log2(gene_length)', 'log2(r_length)', 'log2(r_length/gene_length)', 'log2(distance)','DAR']

HIC_FEATURE_NAMES = [ 'Tanova', '2mo.Q', '9mo.Q', '18mo.Q','9mo-2mo.Q','18mo-9mo.Q', '18mo-2mo.Q',
                     'log2(gene_length)', 'log2(a_length)', 'log2(a_length/gene_length)','Diff_Loop'] #'Qanova', 'Eanova',,'2mo.T', '9mo.T', '18mo.T','9mo-2mo.T', '18mo-9mo.T', '18mo-2mo.T', 

OLD_DEFAULT_DATA_FEATURE_NAMES = {
    'dmr': DMR_FEATURE_NAMES,
    'atac': ATAC_FEATURE_NAMES,
    'hic': HIC_FEATURE_NAMES,
    'mcg_genebody': MCG_GENEBODY_FEATURE_NAMES,
    'mch_genebody': MCH_GENEBODY_FEATURE_NAMES
}

def load_data_old(y_val = "DEG", filepath="data", ct="Oligo_NN", DATA_FEATURE_NAMES=OLD_DEFAULT_DATA_FEATURE_NAMES):
    df = pd.read_csv(f'{filepath}/{ct}/{ct}.luisa_RNA_DEG.csv', index_col=0)
    # df.set_index('gene', inplace=True)

    if y_val == "DEG": 
        gene2value = df[['DEG']]
    elif y_val == "logFC": 
        gene2value = df[['avg_log2FC']]

    DATA = {}

    
    # MCG DMR features
    if 'dmr' in DATA_FEATURE_NAMES.keys():
        DMR_FEATURE_NAMES = DATA_FEATURE_NAMES['dmr']

        skip_bool = True
        try: # Trying to read the filepath for the data
            mcg = pd.read_csv(f'{filepath}/{ct}/{ct}.aDMR_gene.csv')
        except: 
            skip_bool = False
            print(f"No MCG data filepath for this cell type: {ct}, Skipping this data modality")
            print(f"Missig Filepath: {filepath}/{ct}/{ct}.aDMR_gene.csv ")
        if skip_bool: # Calculating all features and the subsetting only for the ones to keep (Can be improved?)
            mcg_feat = mcg
            mcg_feat.rename(columns={'gene_name': 'gene'}, inplace=True)
            mcg_feat['9mo-2mo'] = mcg_feat['9mo'] - mcg_feat['2mo']
            mcg_feat['18mo-9mo'] = mcg_feat['18mo'] - mcg_feat['9mo']
            mcg_feat['18mo-2mo'] = mcg_feat['18mo'] - mcg_feat['2mo']
            mcg_feat['log2(gene_length)'] = np.log2((mcg_feat['gene_end'] - mcg_feat['gene_start']).abs().astype(np.float64))
            mcg_feat['log2(r_length)'] = np.log2((mcg_feat['end'] - mcg_feat['start']).abs().astype(np.float64))
            mcg_feat['log2(r_length/gene_length)'] = np.log2((mcg_feat['end'] - mcg_feat['start'])/(mcg_feat['gene_end'] - mcg_feat['gene_start']))
            mcg_feat['log2(distance)'] = np.log2((mcg_feat['gene_start'] - mcg_feat['start']).abs().astype(np.float64))
            mcg_feat = mcg_feat[['gene', *DMR_FEATURE_NAMES]]
    
            # Checking that the features are valid
            assert mcg_feat.isna().sum().sum() == 0
            assert mcg_feat.isin([np.inf, -np.inf]).sum().sum() == 0
    
            # Adding MCG features to the data dictionary
            DATA['dmr'] = mcg_feat
            print('Processed dmr data')

    # MCG Genebody Features
    if 'mcg_genebody' in DATA_FEATURE_NAMES.keys():
        MCG_GENEBODY_FEATURE_NAMES = DATA_FEATURE_NAMES['mcg_genebody']

        skip_bool = True
        try: # Trying to read the filepath for the data
            genebody = pd.read_csv(f'{filepath}/{ct}/{ct}.mCG_genebody_gene.csv')
        except: 
            skip_bool = False
            print(f"No MCG Genenbody data filepath for this cell type: {ct}, Skipping this data modality")
            print(f"Missig Filepath: {filepath}/{ct}/{ct}.mCG_genebody_gene.csv")
        if skip_bool: # Calculating all features and the subsetting only for the ones to keep (Can be improved?)
            genebody_feat = genebody
            genebody_feat.rename(columns={'gene_name': 'gene'}, inplace=True)
            genebody_feat['9mo-2mo'] = genebody_feat['9mo'] - genebody_feat['2mo']
            genebody_feat['18mo-9mo'] = genebody_feat['18mo'] - genebody_feat['9mo']
            genebody_feat['18mo-2mo'] = genebody_feat['18mo'] - genebody_feat['2mo']
        
            genebody_feat['log2(gene_length)'] = np.log2(genebody_feat['gene_length'])
        
            genebody_feat['(9mo-2mo)*log2(gene_length)'] = genebody_feat['9mo-2mo'] * genebody_feat['log2(gene_length)']
            genebody_feat['(18mo-9mo)*log2(gene_length)'] = genebody_feat['18mo-9mo'] * genebody_feat['log2(gene_length)']
            genebody_feat['(18mo-2mo)*log2(gene_length)'] = genebody_feat['18mo-2mo'] * genebody_feat['log2(gene_length)']
        
            genebody_feat = genebody_feat[['gene', *MCG_GENEBODY_FEATURE_NAMES]]
            genebody_feat= genebody_feat.dropna()
        
            assert genebody_feat.isna().sum().sum() == 0
            assert genebody_feat.isin([np.inf, -np.inf]).sum().sum() == 0
        
            DATA['mcg_genebody'] = genebody_feat
            print('Processed MCG genebody data')

    # MCH Genebody Features
    if 'mch_genebody' in DATA_FEATURE_NAMES.keys():
        MCH_GENEBODY_FEATURE_NAMES = DATA_FEATURE_NAMES['mch_genebody']

        skip_bool = True
        try: # Trying to read the filepath for the data
            genebody = pd.read_csv(f'{filepath}/{ct}/{ct}.mCH_genebody_gene.csv')
        except: 
            skip_bool = False
            print(f"No MCH Genenbody data filepath for this cell type: {ct}, Skipping this data modality")
            print(f"Missig Filepath: {filepath}/{ct}/{ct}.mCH_genebody_gene.csv")
        if skip_bool: # Calculating all features and the subsetting only for the ones to keep (Can be improved?)
            genebody_feat = genebody
            genebody_feat.rename(columns={'gene_name': 'gene'}, inplace=True)
            genebody_feat['9mo-2mo'] = genebody_feat['9mo'] - genebody_feat['2mo']
            genebody_feat['18mo-9mo'] = genebody_feat['18mo'] - genebody_feat['9mo']
            genebody_feat['18mo-2mo'] = genebody_feat['18mo'] - genebody_feat['2mo']
        
            genebody_feat['log2(gene_length)'] = np.log2(genebody_feat['gene_length'])
        
            genebody_feat['(9mo-2mo)*log2(gene_length)'] = genebody_feat['9mo-2mo'] * genebody_feat['log2(gene_length)']
            genebody_feat['(18mo-9mo)*log2(gene_length)'] = genebody_feat['18mo-9mo'] * genebody_feat['log2(gene_length)']
            genebody_feat['(18mo-2mo)*log2(gene_length)'] = genebody_feat['18mo-2mo'] * genebody_feat['log2(gene_length)']
        
            genebody_feat = genebody_feat[['gene', *MCH_GENEBODY_FEATURE_NAMES]]
            genebody_feat= genebody_feat.dropna()
        
            assert genebody_feat.isna().sum().sum() == 0
            assert genebody_feat.isin([np.inf, -np.inf]).sum().sum() == 0
        
            DATA['mch_genebody'] = genebody_feat
            print('Processed MCH genebody data')
    

    # ATAC Seq Features
    if 'atac' in DATA_FEATURE_NAMES.keys():
        ATAC_FEATURE_NAMES = DATA_FEATURE_NAMES['atac']
        
        skip_bool = True
        try: # Trying to read the filepath for the data
            atac = pd.read_csv(f'{filepath}/{ct}/{ct}.peak_gene.csv')
        except: 
            skip_bool = False
            print(f"No ATAC data filepath for this cell type: {ct}, Skipping this data modality")
            print(f"Missig Filepath: {filepath}/{ct}/{ct}.peak_gene.csv")
        if skip_bool: # Calculating all features and the subsetting only for the ones to keep (Can be improved?)
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


    # HiC Features
    if 'hic' in DATA_FEATURE_NAMES.keys():
        HIC_FEATURE_NAMES = DATA_FEATURE_NAMES['hic']

        skip_bool = True
        try: # Trying to read the filepath for the data
            hic = pd.read_csv(f'{filepath}/{ct}/{ct}.Loop_gene.csv.gz')
        except: 
            skip_bool = False
            print(f"No HiC data filepath for this cell type: {ct}, Skipping this data modality")
            print(f"Missig Filepath: {filepath}/{ct}/{ct}.Loop_gene.csv.gz")
        if skip_bool: 
            hic.columns = ['chrom','anchor1_start','anchor1_end','chrom','anchor2_start','anchor2_end',
                        'Qanova','Eanova','Tanova','2mo.Q','9mo.Q','18mo.Q','2mo.T','9mo.T','18mo.T','Diff_Loop',
                        'gene_chrom','gene_start','gene_end','gene_id','strand','gene_name','gene_type']
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
        list_feat = features.groupby('gene').apply(lambda x: x[feature_names].values.tolist(), include_groups=False)
        list_feat = list_feat.reindex(index_order, fill_value=[[0] * len(feature_names)])
        X[feature_type] = list_feat.values.tolist()


    if y_val == "DEG": 
        y = gene2value['DEG'].values.tolist()
        genes = gene2value.index.values.tolist()
        y = np.array([int(i) for i in y])
        Y = gene2value['DEG'].astype(int)
    elif y_val == "logFC": 
        y = gene2value['log2(old/young)'].values.tolist()
        genes = gene2value.index.values.tolist()
        y = np.array([float(i) for i in y])
        Y = gene2value['log2(old/young)'].astype(float)

    return {
        'y': Y,
        'X': X,
    }

