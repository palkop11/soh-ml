import re
from pathlib import Path
from glob import glob

import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold


def make_batteries_info(datadir):
    batteries_paths = glob(datadir + '/**/*/')
    IDs = [Path(s).stem for s in batteries_paths]
    df = pd.DataFrame({'ID':IDs, 'battery_path': batteries_paths})
    df['large_small'] = df.ID.apply(lambda s: re.sub(r'_.*$', '', s))
    df['chem'] = df.ID.apply(lambda s: re.sub(r'^.*_([A-Za-z]+)\d+$', r'\1', s))

    info_csv = pd.read_csv(datadir + '/info.csv').rename(columns = {'Unnamed: 0': 'ID_without_size'})
    info_csv['ID'] = info_csv['large'].apply(lambda s: 'large' if s else 'small') + '_' + info_csv['ID_without_size']
    info_csv[['ID', 'n_cycles']]
    df_merged = pd.merge(df, info_csv[['ID', 'n_cycles']], on = 'ID', how = 'inner')

    return df_merged

SPLITS = {
    'blacklist': [
            'large_LFP10',
            'large_LFP11',
            'large_LTO11',
            'large_LTO12',
            'small_LTO7',
            'small_LTO6',
            'small_LTO4',
            'small_LTO5',
        ],

    'small': [
            'small_LFP4',
            'small_LFP1',
            'small_LFP8',
            'small_LFP5',
            'small_NMC14',
            'small_NMC15',
            'small_NMC11',
            'small_NMC10',
        ],

    'train': [
        'large_LFP13',
        'large_LFP2',
        'large_LFP3',
        'large_NMC12',
        'large_NMC13',
        ],

    'val': [
        'large_LFP6',
        'large_LFP7',
        'large_LTO3',
        ],

    'test': [
        'large_LFP12',
        'large_NMC6',
        'large_NMC7',
        ],
}

twins = [
    ('large_NMC13', 'large_NMC12'),
    ('large_NMC7', 'large_NMC6'),
    ('large_LFP3', 'large_LFP2'),
    ('large_LFP6', 'large_LFP7'),
]

# -------------------------
# get subset info dataframe
# -------------------------

def get_subset_info(subset, datadir, splits = SPLITS):
    info = make_batteries_info(datadir)

    # subset is str
    if isinstance(subset, str):
        if subset in splits.keys():
            subset_list = splits[subset]
            return info.query('ID in @subset_list')
        else:
            raise KeyError(f'Key \'{subset}\' not found')

    # subset is list of IDs
    if isinstance(subset, list):
        return info.query('ID in @subset')

def check_intersec(subset1, subset2):
    intersec = set(subset1).intersection(set(subset2))
    return len(intersec) > 0

# -------------------------
# regular/stratified k-fold
# -------------------------

def create_folds(
    df,
    n_splits=2,
    verbose=False,
    method='regular',
    strat_label=None,
):
    """
    Unified function for creating regular or stratified K-Fold splits.
    
    Parameters:
        df (pd.DataFrame): Input dataframe containing 'ID' column and data.
        n_splits (int): Number of folds to generate.
        verbose (bool): Whether to print fold details.
        method (str): 'regular' for KFold or 'stratified' for StratifiedKFold.
        strat_label (str): Column name (common choice - 'chem') for stratification labels 
        (required if method='stratified').
        
    Returns:
        list: List of dictionaries with 'train' and 'val' IDs for each fold.
    """
    # Validate n_splits
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2 for meaningful splitting")
    
    # Validate method
    if method not in ['regular', 'stratified']:
        raise ValueError("Method must be 'regular' or 'stratified'")
    
    # Validate strat_label for stratified method
    if method == 'stratified':
        if strat_label is None:
            raise ValueError("strat_label must be provided for stratified method")
        if strat_label not in df.columns:
            raise ValueError(f"Column '{strat_label}' not found in dataframe")
    
    # Initialize splitter
    if method == 'stratified':
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        y = df[strat_label].tolist()
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        y = None
    
    X = df.index.to_list()  # Split using dataframe indices
    folds_output = []
    
    # Generate folds
    for fold, (train_idx, val_idx) in enumerate(splitter.split(X, y), 1):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        fold_data = {
            'train': train_df['ID'].tolist(),
            'val': val_df['ID'].tolist()
        }
        folds_output.append(fold_data)
        
        if verbose:
            print(f"Fold {fold}:")
            print("Train IDs:", fold_data['train'])
            print("Val IDs:", fold_data['val'])
            print("\n")
    
    return folds_output

if __name__ == '__main__':

    # check if subsets in SPLITS interesect
    for key1 in SPLITS.keys():
        for key2 in SPLITS.keys():
            if key1 != key2:
                assert not check_intersec(SPLITS[key1], SPLITS[key2]), f'{key1} intersects {key2}!'
    print('There are no intersections between subsets in SPLITS\n')

    # test k-fold
    print('\nregular folds on small subset:')
    regular_folds_small = create_folds(
        df = get_subset_info('small', './DATA/dataset_v5_ts_npz'),
        n_splits=4,
        verbose=True,
    )

    print('\nstrat folds on small subset:')
    strat_folds_small = create_folds(
        df = get_subset_info('small', './DATA/dataset_v5_ts_npz'),
        n_splits=4,
        verbose=True,
        method='stratified',
        strat_label='chem',
    )

    print('\nregular folds on train subset:')
    regular_folds_train = create_folds(
        df = get_subset_info('train', './DATA/dataset_v5_ts_npz'),
        n_splits=5,
        verbose=True,
    )