# TEST CONFIGURATION for single experiment
test_config = {
    'experiment_name': 'testing_on_small_segmented', # also used for TensorBoard logging
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
        'segment_params': {
            'segment_length': 379,
            'overlap': 0, 
            'drop_last':True,
        }
    },

    'model': {
        'input_size': 2,

        'cnn_channels': [4, 16, 64],  # Last element can be final channel dim
        'cnn_kernel_sizes': [3, 3, 3], # None for default
        'cnn_strides': [1, 2, 2], # None for default
        'cnn_paddings': [1, 1, 1], # None for default
        'cnn_use_maxpool': [False, True, True],  # List of booleans for maxpool per block

        'lstm_hidden_size': 64,
        'num_layers': 2,

        'output_size': 1,
        'dropout_prob': 0.4,
        'regressor_hidden_dim': 1024,
        'output_activation': 'sigmoid',
    },

    'training': {
        'resume_ckpt': None,
        'best_model_ckpt': {
            'monitor': 'val_loss',
            'mode': 'min',
        },
        'batch_size': 32,
        'learning_rate': 1e-3,
        'scheduler_parameters': {
            'scheduler_type': None,
            'plateau_factor': 0.5,
            'plateau_patience': 5,
            'cosine_t_max': 10,
        },
        'loss_type': 'huber1.0',
        'epochs': 50,
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