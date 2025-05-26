import argparse
import yaml
from pathlib import Path
from modules.experiment import BatteryExperiment

# TEST CONFIGURATION
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
        'resume_ckpt': 'from_last',
        'batch_size': 32,
        'learning_rate': 1e-3,
        'scheduler': 'reduce_on_plateu',
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
    }
}

def load_config(config_path=None):
    """Load configuration from file or use default"""
    if config_path:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Loaded configuration from {config_path}")
    else:
        config = test_config
        print("Using test_config")
    return config

def run_experiment(
        config: str | dict, 
        savefig: bool = True,
        ):
    """
    config may be str (path to config in .yaml file)
    or config in python dictionary
    or None (use test_config in this case)
    """

    # already prepared config 
    if isinstance(config, dict):
        experiment_config = config
    # load from file or use test_config    
    else:
        experiment_config = load_config(config)    
    
    # Run experiment
    experiment = BatteryExperiment(experiment_config)
    results = experiment.run()

    # Generate analysis plots
    experiment.analyze_results()
    print('metrics on train:')
    train_metrics = experiment.calculate_metrics('train')
    print('metrics on val:')
    val_metrics = experiment.calculate_metrics('val')

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run battery experiment')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to YAML config file (optional)')
    args = parser.parse_args()
    run_experiment(args.config)

if __name__ == '__main__':
    main()