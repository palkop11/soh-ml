import argparse
import yaml
from pathlib import Path
from modules.experiment import BatteryExperiment

# Default configuration (same as original)
DEFAULT_CONFIG = {
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
    'loss_type': 'huber',
    
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

def load_config(config_path=None):
    """Load configuration from file or use default"""
    if config_path:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Loaded configuration from {config_path}")
    else:
        config = DEFAULT_CONFIG
        print("Using default configuration")
    return config

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run battery experiment')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to YAML config file (optional)')
    args = parser.parse_args()

    # Load configuration
    experiment_config = load_config(args.config)

    # Run experiment
    experiment = BatteryExperiment(experiment_config)
    results = experiment.run()

    # Generate analysis plots
    _ = experiment.analyze_results('train', savefig=True)
    _ = experiment.analyze_results('val', savefig=True)
    print('metrics on train:')
    train_metrics = experiment.calculate_metrics('train')
    print('metrics on val:')
    val_metrics = experiment.calculate_metrics('val')

if __name__ == '__main__':
    main()