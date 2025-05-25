import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import yaml
from pathlib import Path

import numpy as np
import random
import matplotlib.pyplot as plt

from .data_splitting import get_subset_info
from .datasets import DataSetCreation
from .learning_pipeline import BatteryDataModule, BatteryPipeline, collate_fn
from .models import UnifiedBatteryModel

class BatteryExperiment:
    def __init__(self, config):
        self.config = config
        self.set_seed()
        self.logger = self.create_logger()
        self.prepare_paths()
        self.model = None
        self.datamodule = None
        self.pipeline = None
        self.trainer = None

    def set_seed(self, seed=None):
        seed = seed or self.config['seed']
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def prepare_paths(self):
        # Get the version from the logger (auto-generated if not specified)
        version = self.logger.version if isinstance(self.logger.version, str) else f"version_{self.logger.version}"
        
        # Versioned experiment directory
        self.exp_dir = Path(self.config['logging']['log_dir']) / self.config['experiment_name'] / version
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Set name for config in .yaml
        self.config_filename = '_'.join([self.config['experiment_name'], version, 'config']) + '.yaml'

        # Checkpoints inside versioned dir
        self.checkpoint_dir = self.exp_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)

    def create_logger(self):
        return TensorBoardLogger(
            save_dir=self.config['logging']['log_dir'],
            name=self.config['experiment_name'],
            version=None, #self.config.get('version', None)
        )

    def prepare_datasets(self):
        # Train dataset
        train_info = get_subset_info(
            subset = self.config['data']['train'],
            datadir = self.config['data']['datadir']
        )
        self.train_ds = DataSetCreation(
            train_info,
            fit_normalization=True,
            normalization_types=self.config['data']['normalization'],
            n_diff=self.config['data']['n_diff']
        )

        # Validation dataset
        val_info = get_subset_info(
            subset = self.config['data']['val'],
            datadir = self.config['data']['datadir']
        )
        self.val_ds = DataSetCreation(
            val_info,
            normalize=self.train_ds.normalize,
            n_diff=self.config['data']['n_diff']
        )

        # Test dataset (optional)
        if self.config['data'].get('test'):
            test_info = get_subset_info(
                subset = self.config['data']['test'],
                datadir = self.config['data']['datadir']
            )
            self.test_ds = DataSetCreation(
                test_info,
                normalize=self.train_ds.normalize,
                n_diff=self.config['data']['n_diff']
            )

    def create_model(self):
        model_config = self.config['model']
        
        return UnifiedBatteryModel(
            input_size=model_config['input_size'],
            cnn_hidden_dim=model_config['cnn_hidden_dim'],
            cnn_channels=model_config['cnn_channels'],
            lstm_hidden_size=model_config['lstm_hidden_size'],
            num_layers=model_config['num_layers'],
            output_size=model_config['output_size'],
            dropout_prob=model_config['dropout'],
            regressor_hidden_dim=model_config['regressor_hidden_dim'],
            output_activation=model_config['output_activation'],
        )

    def create_datamodule(self):
        return BatteryDataModule(
            train_dataset=self.train_ds.dataset,
            val_dataset=self.val_ds.dataset,
            test_dataset=self.test_ds.dataset if hasattr(self, 'test_ds') else None,
            batch_size=self.config['training']['batch_size'],
            collate_fn=collate_fn
        )

    def create_pipeline(self):
        return BatteryPipeline(
            model=self.model,
            denormalize_y=self.train_ds.denormalize['y'],
            loss_type=self.config['training']['loss_type'], # 'mse', 'huber', 'bce'
            learning_rate=self.config['training']['learning_rate'],
            metrics = self.config['metrics'], # list like ['mse', 'mae', 'mape', 'r2', 'rcc']
        )

    def create_trainer(self):
        return pl.Trainer(
            max_epochs=self.config['training']['epochs'],
            logger=self.logger,
            accelerator=self.config['training']['accelerator'],
            devices=self.config['training']['devices'],
            callbacks=[
                pl.callbacks.ModelCheckpoint(
                    dirpath=self.checkpoint_dir,  # Now inside versioned dir
                    monitor='val_loss',
                    mode='min',
                    save_top_k=1
                )
            ],
            enable_progress_bar=self.config['logging']['progress_bar'],
            deterministic=True
        )

    def run(self):
        # Save config for reproducibility
        with open(self.exp_dir / self.config_filename, 'w') as f:
            yaml.dump(self.config, f)

        # Experiment pipeline
        self.prepare_datasets()
        self.model = self.create_model()
        self.datamodule = self.create_datamodule()
        self.pipeline = self.create_pipeline()
        self.trainer = self.create_trainer()

        # Checkpoint path (specify in config or auto-detect latest)
        ckpt_path = self.config['training']['resume_ckpt']
        if ckpt_path == 'auto':
            # Find latest checkpoint in checkpoint_dir
            checkpoints = list(self.checkpoint_dir.glob('*.ckpt'))
            if checkpoints:
                ckpt_path = max(checkpoints, key=lambda x: x.stat().st_ctime)
            else:
                ckpt_path = None
        elif ckpt_path:
            ckpt_path = Path(ckpt_path)
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Checkpoint {ckpt_path} not found!")

        # Training (resumes from checkpoint if provided)
        self.trainer.fit(
            self.pipeline, 
            datamodule=self.datamodule, 
            ckpt_path=str(ckpt_path) if ckpt_path else None
        )

        # Testing (unchanged)
        if hasattr(self.datamodule, 'test_dataset'):
            test_results = self.trainer.test(datamodule=self.datamodule)
            return test_results
        return None

    def calculate_metrics(self, dataloader = None):
        if dataloader is None:
            dataloader = self.datamodule.val_dataloader()
        elif isinstance(dataloader, str):
            available_dataloaders = {
                'train': self.datamodule.train_dataloader(),
                'val': self.datamodule.val_dataloader(),
                'test': self.datamodule.test_dataloader(),
                }
            if dataloader in available_dataloaders.keys():
                dataloader = available_dataloaders[dataloader]
            else:
                print(f'no such dataloader {dataloader}')
                return None

        return self.trainer.validate(
            model = self.pipeline,
            dataloaders = dataloader, )

    def analyze_results(
        self, 
        datasets = ['train', 'val'],
        plot=False,  
        savefig=True, 
        xylims=None,
    ):
        """Generate predictions and plots for specified subsets 
        (train and val by default) in subplots"""
        plot = self.config['logging']['plot']
        savefig = self.config['logging']['savefig']

        self.pipeline.eval()
        results = {}

        for dataset in datasets:
            # Get the appropriate dataloader
            dataloader = getattr(self.datamodule, f'{dataset}_dataloader')()
            
            # Generate predictions
            y_true, y_pred = [], []
            with torch.no_grad():
                for batch in dataloader:
                    x = batch['x'].to(self.pipeline.device)
                    lengths = batch['lengths'].to(self.pipeline.device)
                    preds = self.pipeline(x, lengths)
                    y_true.append(batch['y'].cpu())
                    y_pred.append(preds.cpu())

            y_true = torch.cat(y_true)
            y_pred = torch.cat(y_pred)

            # Denormalize using training dataset stats
            y_true_denorm = self.train_ds.denormalize['y'](y_true)
            y_pred_denorm = self.train_ds.denormalize['y'](y_pred)

            # Calculate metrics
            metrics = {}
            for name, metric_cls in self.pipeline.metric_classes.items():
                metric = metric_cls().to(y_pred_denorm.device)
                metrics[name] = metric(y_pred_denorm, y_true_denorm).item()

            results[dataset] = {
                'y_true': y_true_denorm,
                'y_pred': y_pred_denorm,
                'metrics': metrics
            }

        # Create figure with subplots
        fig, axes = plt.subplots(nrows=1, ncols=len(datasets), figsize=(20, 8))
        axes = axes.flatten()

        for ax, dataset in zip(axes, datasets):
            data = results[dataset]
            y_true_denorm = data['y_true']
            y_pred_denorm = data['y_pred']
            metrics = data['metrics']

            # Scatter plot
            ax.scatter(y_true_denorm, y_pred_denorm, alpha=0.5)
            ax.plot([y_true_denorm.min(), y_true_denorm.max()],
                    [y_true_denorm.min(), y_true_denorm.max()], 'r--')
            ax.set_xlabel('True Values')
            ax.set_ylabel('Predictions')
            ax.set_title(f'{dataset.capitalize()} Set: True vs Predicted')

            # Add metrics text
            metric_text = "\n".join([f"{name.upper()}: {val:.4f}" for name, val in metrics.items()])
            ax.text(0.05, 0.95, metric_text, transform=ax.transAxes,
                    verticalalignment='top', horizontalalignment='left',
                    bbox=dict(facecolor='white', alpha=0.5))

            # Apply axis limits if specified
            if xylims is not None:
                ax.set_xlim(xylims)
                ax.set_ylim(xylims)

        # Add experiment info
        exp_name = self.config['experiment_name']
        version = getattr(self.logger, 'version', 'unknown')
        fig.suptitle(
            f"Experiment: {exp_name}\n"
            f"Version: {version}",
            y=1.02, 
            fontsize=10,
            ha='left',
            x=0.15
        )

        plt.tight_layout()

        # Save and show options
        if savefig:
            fig_filename = f"{exp_name}_version_{version}_{'_'.join(datasets)}.png"
            plt.savefig(
                self.exp_dir / fig_filename,
                bbox_inches='tight',
                dpi=300
            )
        if plot:
            plt.show()
        else:
            plt.close()