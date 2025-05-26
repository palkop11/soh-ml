import pandas as pd
from pathlib import Path
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np

def parse_tb_events(log_dir):
    """Parse TensorBoard event files from a directory."""
    try:
        event_acc = EventAccumulator(str(log_dir))
        event_acc.Reload()
        return {
            tag: [e.value for e in event_acc.Scalars(tag)]
            for tag in event_acc.Tags()['scalars']
        }
    except Exception as e:
        print(f"Error parsing TensorBoard events in {log_dir}: {e}")
        return {}

def get_final_metrics(tb_data):
    """Extract final metrics from the last recorded epoch."""
    metrics = {}
    for tag in tb_data:
        if tag.startswith('val_'):
            metrics[tag] = tb_data[tag][-1] if tb_data[tag] else np.nan
    return metrics

def load_config(search_dir):
    """Search for *_config.yaml file inside a directory recursively."""
    config_files = list(search_dir.rglob("*_config.yaml"))
    if not config_files:
        print(f"No config file found in: {search_dir}")
        return None
    try:
        with open(config_files[0], 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config in {search_dir}: {e}")
        return None

def extract_params(config):
    """Extract parameters from config with type normalization."""
    params = {}
    for section in ['model', 'training', 'data']:
        if section in config:
            for k, v in config[section].items():
                params[f"{section}_{k}"] = tuple(v) if isinstance(v, list) else v
    return params

def process_version_dir(version_dir):
    """Process a single version directory containing TensorBoard logs."""
    version_dir = Path(version_dir)
    print(f"Processing: {version_dir}")
    
    config = load_config(version_dir)
    params = extract_params(config) if config else {}
    
    tb_data = parse_tb_events(version_dir)
    metrics = get_final_metrics(tb_data)

    experiment_data = {
        'experiment_name': version_dir.parent.name,
        'version': version_dir.name,
        'path': str(version_dir.parent),
        'version_path': str(version_dir),
        **params,
        **metrics
    }

    # Add cross-validation info if detected in folder name
    if '_fold_' in version_dir.parent.name:
        parts = version_dir.parent.name.split('_fold_')
        experiment_data.update({
            'param_hash': parts[0].split('_')[-1],
            'fold_number': int(parts[1])
        })

    return experiment_data

def analyze_logs(master_dir, cv_mode='auto'):
    """Main analysis function with CV detection."""
    master_dir = Path(master_dir)
    experiments = []

    for version_dir in master_dir.rglob("version_*"):
        if version_dir.is_dir():
            result = process_version_dir(version_dir)
            if result:
                experiments.append(result)

    if not experiments:
        print("No valid experiments found")
        return pd.DataFrame()

    df = pd.DataFrame(experiments)

    # Auto-detect CV mode
    if cv_mode == 'auto':
        cv_mode = 'cv' if 'fold_number' in df.columns else 'single'

    return aggregate_cv_results(df) if cv_mode == 'cv' else format_single_experiments(df)

def aggregate_cv_results(df):
    """Aggregate cross-validation results."""
    if 'param_hash' not in df.columns:
        print("No CV experiments found - switching to single mode")
        return format_single_experiments(df)

    metric_cols = [c for c in df.columns if c.startswith('val_')]
    grouped = df.groupby('param_hash', group_keys=False)

    agg_funcs = {metric: ['mean', 'std', 'min', 'max', 'count'] for metric in metric_cols}
    agg_df = grouped.agg(agg_funcs)
    agg_df.columns = [f"{metric}_{stat}" for metric, stat in agg_df.columns]

    hyperparams = df.drop(columns=metric_cols + ['fold_number', 'path', 'version', 'version_path']
                         ).drop_duplicates('param_hash')

    return pd.merge(hyperparams, agg_df, on='param_hash')

def format_single_experiments(df):
    """Format individual experiment results."""
    sort_col = 'val_loss' if 'val_loss' in df.columns else 'experiment_name'
    df = df.sort_values(sort_col)
    keep_cols = ['experiment_name', 'version'] + \
                [c for c in df.columns if c.startswith(('model_', 'training_', 'data_', 'val_'))]
    return df[keep_cols].reset_index(drop=True)