# TEST CONFIGURATION for single experiment
test_config = {
    'experiment_name': 'testing_on_small', # also used for TensorBoard logging
    'seed': 42,

    'data': {
        'datadir':'./DATA/dataset_v5_ts_npz/',
        'train': [
                    'small_LFP4',
                    'small_LFP8',
                    'small_LFP1',
                    'small_NMC10',
                    'small_NMC14',
                    'small_NMC15',
                ],
        'val': ['small_LFP5', 'small_NMC11'],
        #'test': None,
        'normalization': {'x': None, 'y': 'minmax_zero_one'},
        'n_diff': 0,
    },

    'model': {
        'input_size': 2,
        'cnn_hidden_dim': 32,
        'cnn_channels': [4, 8, 16],
        'lstm_hidden_size': 32,
        'num_layers': 1,
        'output_size': 1,
        'dropout': 0.,
        'regressor_hidden_dim': 1024,
        'output_activation': 'sigmoid',
    },

    'training': {
        'resume_ckpt': None,
        'batch_size': 32,
        'learning_rate': 1e-3,
        'scheduler': 'reduce_on_plateu',
        'loss_type': 'huber1.0',
        'epochs': 1,
        'accelerator': 'auto',
        'devices': 1,
    },

    'metrics': 'all',

    'logging': {
        'log_dir': './LOGS',
        'progress_bar': True,
        'plot': False,
        'savefig': True,
        'torchinfo_model_summary': True,
    }
}

# TEST CONFIGURATION for cv-experiment
cv_test_config_dict = {
    'master_name': 'cv_test_experiment',
    'base_config': test_config,
    'hyperparam_grid': {
        'model': {
            'cnn_hidden_dim': [16, 32],
        },
    },
    'crossval_settings': {
        'n_splits': 2,
        'method': 'stratified',
        'strat_label': 'chem',
        'dataset_subset': 'small',
    }
}