from modules.experiment import BatteryExperiment

experiment_config = {
    'experiment_name': 'testing_on_small', # also used for TensorBoard logging
    'seed': 42,

    'data': {
        'datadir':'./DATA/dataset_v5_ts_npz/',
        'train': ['small_NMC15', 'small_NMC11', 'small_LFP5', 'small_LFP8'],
        'val': ['small_NMC10', 'small_NMC14', 'small_LFP4', 'small_LFP1'],
        #'test': None, # comment this line if you do not pass test dataset
        'normalization': {'x': None, 'y': 'minmax_zero_one'},
        'n_diff': 0,
    },

    'model': {
        'input_size': 2,
        'cnn_hidden': 32,
        'lstm_hidden_size': 2,
        'num_layers': 1,
        'output_size': 1,
        'dropout': 0.,
        'regressor_hidden_dim': None,
        'output_activation': 'sigmoid',
    },

    'resume_ckpt': 'auto',
    'loss_type': 'bce',
    
    'training': {
        'batch_size': 8,
        'learning_rate': 1e-3,
        'epochs': 2,
        'accelerator': 'auto',
        'devices': 1,
    },

    'logging': {
        'log_dir': './LOGS',
        'progress_bar': True,
    }
}

# Run experiment
experiment = BatteryExperiment(experiment_config)
results = experiment.run()

# Generate analysis plots
_ = experiment.analyze_results('train', savefig = True)
_ = experiment.analyze_results('val', savefig = True)
print('metrics on train:')
train_metrics = experiment.calculate_metrics('train')
print('metrics on val:')
val_metrics = experiment.calculate_metrics('val')