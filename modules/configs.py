# TEST CONFIGURATION for single experiment
test_config = {
    'experiment_name': 'testing_on_small_64', # also used for TensorBoard logging
    'seed': 42,

    'precision': 'float64',
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
        'segment_params': {
            'segment_length': None,
            'overlap': 0, 
            'drop_last':True,
        }
    },

    'model': {
        'input_size': 2,

        'cnn_channels': [8, 16, 32, 64],  # Last element can be final channel dim
        'cnn_kernel_sizes': [3, 5, 7, 7], # None for default
        'cnn_strides': [1, 2, 2, 2], # None for default
        'cnn_paddings': [1, 1, 1, 2], # None for default
        'cnn_use_maxpool': [False, True, True, True],  # List of booleans for maxpool per block

        'lstm_hidden_size': 128,
        'num_layers': 1,

        'output_size': 1,
        'dropout_prob': 0.25,
        'regressor_hidden_dim': 256,
        'output_activation': 'sigmoid',
    },

    'training': {
        'resume_ckpt': None,
        'best_model_ckpt': [
            {
                'monitor': 'val_loss',
                'mode': 'min',
            },
            {
                'monitor': 'val_pcc',
                'mode': 'max',
            },
            {
                'monitor': 'val_r2',
                'mode': 'max',
            },
        ],
        'batch_size': 32,
        'learning_rate': 1e-3,
        'scheduler_parameters': {
            'scheduler_type': 'cosine',
            'plateau_factor': 0.5,
            'plateau_patience': 5,
            'cosine_t_max': 25,
        },
        'loss_type': 'huber1.0',
        'epochs': 5,
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
            'dropout_prob': [0., 0.25],
        },
    },
    'crossval_settings': {
        'n_splits': 2,
        'method': 'stratified',
        'strat_label': 'chem',
        'dataset_subset': 'small',
    }
}