import re
from pathlib import Path
from glob import glob

import pandas as pd


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

BLACKLIST = [
        'large_LFP10',
        'large_LFP11',
        'large_LTO11',
        'large_LTO12',
        'small_LTO7',
        'small_LTO6',
        'small_LTO4',
        'small_LTO5',
    ]

SMALL_LIST = [
        'small_LFP4',
        'small_LFP1',
        'small_LFP8',
        'small_LFP5',
        'small_NMC14',
        'small_NMC15',
        'small_NMC11',
        'small_NMC10',
    ]

'''
twins:
    large_NMC13, large_NMC12
    large_NMC7, large_NMC6
    large_LFP3, large_LFP2
    large_LFP6, large_LFP7
'''

TRAIN_VAL_TEST = {
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

def get_subset_info(names, datadir):
    info = make_batteries_info(datadir)
    if isinstance(names, str):
        match names:
            case "blacklist":
                return info.query('ID in @BLACKLIST')
            case "small":
                return info.query('ID in @SMALL_LIST')
            case ["train", "val", "test"]:
                id_in_info = TRAIN_VAL_TEST[names]
                return info.query('ID in @id_in_info')

    # name is 
    if isinstance(names, list):
        return info.query('ID in @names') 

if __name__ == '__main__':

    def check_intersec(subset1, subset2):
        intersec = set(subset1).intersection(set(subset2))
        return len(intersec) > 0
    
    for key in TRAIN_VAL_TEST:
        assert not check_intersec(TRAIN_VAL_TEST[key], BLACKLIST), f'{key} intersects blacklist!'

    for key in TRAIN_VAL_TEST:
        assert not check_intersec(TRAIN_VAL_TEST[key], BLACKLIST), f'{key} intersects small_list!'

    assert not check_intersec(BLACKLIST, SMALL_LIST), 'blacklist intersects small_list!'
    assert not check_intersec(TRAIN_VAL_TEST['train'], TRAIN_VAL_TEST['val']), 'train intersects val!'
    assert not check_intersec(TRAIN_VAL_TEST['train'], TRAIN_VAL_TEST['test']), 'train intersects val!'

    print('There are no intersections between blacklist, small_set, train, val, test!')