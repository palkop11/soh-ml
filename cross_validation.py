# [file name]: cross_validation.py
import json
import hashlib
import yaml
from pathlib import Path
import copy
import itertools
from typing import Dict, Any, Union, Optional
from datetime import datetime

from modules.data_splitting import get_subset_info, create_folds
from modules.experiment import BatteryExperiment

class CrossValidator:
    def __init__(
        self,
        config: Union[str, Dict[str, Any]],  # Accept both path and dict
        progress_file: str = "cv_progress.json",
        resume: bool = True
    ):
        """
        Initialize cross-validator with configuration
        
        Args:
            config: Either path to YAML config file or config dictionary
            progress_file: Path to store/load progress
            resume: Whether to resume from previous progress
        """
        self.config = self._load_config(config)
        self.progress_file = Path(progress_file)
        self.resume = resume
        self.progress = self._init_progress()

    def _load_config(self, config: Union[str, Dict]) -> Dict[str, Any]:
        """Load and validate configuration from file or dict"""
        if isinstance(config, str):
            with open(config) as f:
                cfg = yaml.safe_load(f)
        elif isinstance(config, dict):
            cfg = copy.deepcopy(config)
        else:
            raise TypeError("Config must be file path or dictionary")
        
        return self._validate_config(cfg)

    def _validate_config(self, config: Dict) -> Dict:
        """Validate configuration structure"""
        required_sections = ['base_config', 'hyperparam_grid', 'crossval_settings']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required section '{section}' in config")
        
        # Ensure base config has required structure
        base = config['base_config']
        for section in ['data', 'model', 'training']:
            if section not in base:
                raise ValueError(f"base_config missing required section '{section}'")
        
        return config

    def _init_progress(self) -> Dict[str, bool]:
        """Initialize progress tracking"""
        if self.resume and self.progress_file.exists():
            return json.loads(self.progress_file.read_text())
        return {}

    def _hash_params(self, params: Dict[str, Any]) -> str:
        """Generate unique hash for parameter combination"""
        return hashlib.md5(
            json.dumps(params, sort_keys=True).encode()
        ).hexdigest()[:8]

    def _generate_param_combinations(self) -> list:
        """Generate parameter combinations from grid"""
        param_grid = self.config['hyperparam_grid']
        sections = []
        values = []
        
        for section, params in param_grid.items():
            for param, vals in params.items():
                sections.append((section, param))
                values.append(vals if isinstance(vals, list) else [vals])
        
        combinations = []
        for combo in itertools.product(*values):
            param_dict = {}
            for (section, param), value in zip(sections, combo):
                param_dict.setdefault(section, {})[param] = value
            combinations.append(param_dict)
        
        return combinations

    def _create_folds(self, data_info) -> list:
        """Create cross-validation folds"""
        cv_settings = self.config['crossval_settings']
        return create_folds(
            df=data_info,
            n_splits=cv_settings.get('n_splits', 5),
            method=cv_settings.get('method', 'stratified'),
            strat_label=cv_settings.get('strat_label', 'chem'),
            verbose=cv_settings.get('verbose', False)
        )

    def _get_experiment_id(self, param_hash: str, fold_idx: int) -> str:
        """Generate unique experiment identifier"""
        return f"{param_hash}_fold_{fold_idx}"

    def _should_skip(self, experiment_id: str) -> bool:
        """Check if experiment should be skipped"""
        return self.progress.get(experiment_id, False)

    def _update_config_for_fold(
        self,
        base_config: Dict[str, Any],
        params: Dict[str, Any],
        fold: Dict[str, list]
    ) -> Dict[str, Any]:
        """Create experiment config for specific fold"""
        config = copy.deepcopy(base_config)
        
        # Merge hyperparameters
        for section, section_params in params.items():
            config[section].update(section_params)
        
        # Update data splits
        config['data']['train'] = fold['train']
        config['data']['val'] = fold['val']
        
        # Add crossval metadata
        config['crossval'] = {
            'param_hash': self._hash_params(params),
            'start_time': datetime.now().isoformat()
        }
        
        return config

    def run(self):
        """Execute full cross-validation process"""
        # Prepare base configuration
        base_config = self.config['base_config']
        data_info = get_subset_info(
            subset=self.config['crossval_settings'].get('dataset_subset', 'train'),
            datadir=base_config['data']['datadir']
        )
        
        # Generate all components
        param_combinations = self._generate_param_combinations()
        folds = self._create_folds(data_info)
        
        # Main cross-validation loop
        for params in param_combinations:
            param_hash = self._hash_params(params)
            
            for fold_idx, fold in enumerate(folds):
                experiment_id = self._get_experiment_id(param_hash, fold_idx)
                
                if self._should_skip(experiment_id):
                    print(f"Skipping completed experiment: {experiment_id}")
                    continue
                
                try:
                    # Configure experiment
                    experiment_config = self._update_config_for_fold(
                        base_config, params, fold
                    )
                    experiment_config['version'] = experiment_id
                    experiment_config['resume_ckpt'] = 'auto'
                    
                    # Run experiment
                    experiment = BatteryExperiment(experiment_config)
                    experiment.run()
                    
                    # Record success
                    self.progress[experiment_id] = True
                except Exception as e:
                    print(f"Error in {experiment_id}: {str(e)}")
                    self.progress[experiment_id] = False
                
                # Save progress after each experiment
                self._save_progress()

    def _save_progress(self):
        """Save progress to file with atomic write"""
        temp_file = self.progress_file.with_suffix('.tmp')
        with open(temp_file, 'w') as f:
            json.dump(self.progress, f, indent=2)
        temp_file.rename(self.progress_file)

# Usage examples
if __name__ == '__main__':
    # Example 1: From YAML config file
    import argparse
    
    parser = argparse.ArgumentParser(description='Run cross-validation')
    parser.add_argument('--config', type=str,
                       help='Path to YAML config file')
    parser.add_argument('--progress-file', type=str, default='cv_progress.json',
                       help='Path to progress tracking file')
    parser.add_argument('--no-resume', action='store_false', dest='resume',
                       help='Disable resume functionality')
    
    args = parser.parse_args()
    
    if args.config:
        validator = CrossValidator(
            config=args.config,
            progress_file=args.progress_file,
            resume=args.resume
        )
        validator.run()
    else:
        print('config file was not specified in args')

    # Example 2: From Python dictionary
    cv_config_dict = {
        'base_config': {
            'experiment_name': 'cv_from_dict',
            'seed': 42,
            'data': {
                'datadir': './DATA/dataset_v5_ts_npz/',
                'normalization': {'x': None, 'y': 'minmax_zero_one'},
                'n_diff': 0,
            },
            'model': {
                'input_size': 2,
                'lstm_hidden_size': 32,
                'num_layers': 1,
                'output_size': 1,
                'dropout': 0.,
                'regressor_hidden_dim': 1024,
                'output_activation': 'sigmoid',
            },
            'training': {
                'epochs': 50,
                'accelerator': 'auto',
                'devices': 1,
            },
            'logging': {
                'log_dir': './LOGS',
                'progress_bar': True,
            }
        },
        'hyperparam_grid': {
            'training': {
                'learning_rate': [1e-3, 1e-4],
            }
        },
        'crossval_settings': {
            'n_splits': 3,
            'method': 'regular',
        }
    }

    # Uncomment to run from dict
    # validator = CrossValidator(config=cv_config_dict)
    # validator.run()