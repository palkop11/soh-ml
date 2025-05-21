import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import yaml
from pathlib import Path

from .data_splitting import get_subset_info

class BatteryExperiment:
    def __init__(self, config):
        self.config = config
        self.set_seed()
        self.prepare_paths()
        self.logger = self.create_logger()
        self.model = None
        self.datamodule = None
        self.pipeline = None
        self.trainer = None

    def set_seed(self, seed=None):
        seed = seed or self.config.get('seed', 42)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def prepare_paths(self):
        self.exp_dir = Path(self.config['logging']['log_dir']) / self.config['experiment_name']
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.exp_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)

    def create_logger(self):
        return TensorBoardLogger(
            save_dir=self.config['logging']['log_dir'],
            name=self.config['experiment_name'],
            version=self.config.get('version', None)
        )

    def prepare_datasets(self):
        # Train dataset
        train_info = get_subset_info(
            names = self.config['data']['train'],
            datadir = self.config['data']['datadir']
        )
        self.train_ds = DataSetCreation(
            train_info,
            fit_normalization=True,
            normalization_types=self.config['data']['normalization'],
            n_diff=self.config['data'].get('n_diff', None)
        )

        # Validation dataset
        val_info = get_subset_info(
            names = self.config['data']['val'],
            datadir = self.config['data']['datadir']
        )
        self.val_ds = DataSetCreation(
            val_info,
            normalize=self.train_ds.normalize,
            n_diff=self.config['data'].get('n_diff', None)
        )

        # Test dataset (optional)
        if 'test_info' in self.config['data']:
            test_info = get_subset_info(
                names = self.config['data']['test'],
                datadir = self.config['data']['datadir']
            )
            self.test_ds = DataSetCreation(
                test_info,
                normalize=self.train_ds.normalize,
                n_diff=self.config['data'].get('n_diff', None)
            )

    def create_model(self):
        model_config = self.config['model']
        if model_config['type'] == 'LSTM':
            return LSTMModel(
                input_size=model_config['input_size'],
                hidden_size=model_config['hidden_size'],
                num_layers=model_config['num_layers'],
                output_size=model_config['output_size']
            )
        elif model_config['type'] == 'CNN-LSTM':
            return CNNLSTMModel(
                input_size=model_config['input_size'],
                cnn_hidden=model_config['cnn_hidden'],
                lstm_hidden_size=model_config['lstm_hidden_size'],
                num_layers=model_config['num_layers'],
                output_size=model_config['output_size'],
                dropout_prob=model_config.get('dropout', 0.25),
                output_activation=model_config.get('output_activation', 'tanh')
            )
        elif model_config['type'] == 'CNN-LSTM-overfit':
            return CNN_LSTM_overfit_Model(
                input_size=model_config['input_size'],
                cnn_hidden=model_config['cnn_hidden'],
                lstm_hidden_size=model_config['lstm_hidden_size'],
                num_layers=model_config['num_layers'],
                output_size=model_config['output_size'],
                dropout_prob=model_config.get('dropout', 0),
                regressor_hidden_dim = model_config.get('regressor_hidden_dim', 1024),
                output_activation=model_config.get('output_activation', 'tanh')
            )
        else:
            raise ValueError(f"Unknown model type: {model_config['type']}")

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
            loss_fn=torch.nn.MSELoss(),
            learning_rate=self.config['training']['learning_rate']
        )

    def create_trainer(self):
        return pl.Trainer(
            max_epochs=self.config['training']['epochs'],
            logger=self.logger,
            accelerator=self.config['training'].get('accelerator', 'auto'),
            devices=self.config['training'].get('devices', 1),
            callbacks=[
                pl.callbacks.ModelCheckpoint(
                    dirpath=self.checkpoint_dir,
                    monitor='val_loss',
                    mode='min',
                    save_top_k=1
                )
            ],
            enable_progress_bar=self.config['logging'].get('progress_bar', True),
            deterministic=True
        )

    def run(self):
        # Save config for reproducibility
        with open(self.exp_dir / 'config.yaml', 'w') as f:
            yaml.dump(self.config, f)

        # Experiment pipeline
        self.prepare_datasets()
        self.model = self.create_model()
        self.datamodule = self.create_datamodule()
        self.pipeline = self.create_pipeline()
        self.trainer = self.create_trainer()

        # Training
        self.trainer.fit(self.pipeline, datamodule=self.datamodule)

        # Testing
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

    def analyze_results(self, dataset_type='val', savefig = False, xylims = None):
        """Generate predictions and plots for analysis"""
        self.pipeline.eval()
        dataloader = {
            'train': self.datamodule.train_dataloader(),
            'val': self.datamodule.val_dataloader(),
            'test': self.datamodule.test_dataloader()
        }[dataset_type]

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

        # Denormalize
        y_true_denorm = self.train_ds.denormalize['y'](y_true)
        y_pred_denorm = self.train_ds.denormalize['y'](y_pred)

        # Plotting
        plt.figure()
        plt.scatter(y_true_denorm, y_pred_denorm, alpha=0.5)
        plt.plot([y_true_denorm.min(), y_true_denorm.max()],
                 [y_true_denorm.min(), y_true_denorm.max()], 'r--')
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.title(f'{dataset_type.capitalize()} Set: True vs Predicted')
        if xylims is not None:
            plt.xlim(xylims)
            plt.ylim(xylims)
        if savefig:
            plt.savefig(self.exp_dir / f'{dataset_type}_predictions.png')
        plt.show()
        #plt.close()

        return y_true_denorm.numpy(), y_pred_denorm.numpy()