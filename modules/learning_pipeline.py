from torch.nn.utils.rnn import pad_sequence

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics import MeanSquaredError, MeanAbsolutePercentageError, PearsonCorrCoef, R2Score, MetricCollection, MeanAbsoluteError

def collate_fn(batch):
    """Collate function compatible with PyTorch Lightning"""
    # Sort batch by sequence length (descending) for RNN efficiency
    batch.sort(key=lambda x: x['x'].size(0), reverse=True)

    # Extract components
    x = [item['x'] for item in batch]  # List of [seq_len, 2] tensors
    lengths = torch.tensor([len(f) for f in x], dtype=torch.long)
    y = torch.stack([item['y'] for item in batch])

    # Pad sequences to max length in batch
    padded_x = pad_sequence(x, batch_first=True, padding_value=0.0)

    return {
        'x': padded_x,      # Padded x tensor [batch_size, max_seq_len, 2]
        'lengths': lengths, # Sequence lengths tensor [batch_size]
        #'y': y,       # Targets tensor [batch_size]
        'y': y.reshape(-1, 1),
    }

# -----------------
# BatteryDataModule
# -----------------

class BatteryDataModule(pl.LightningDataModule):
    def __init__(
            self,
            train_dataset,
            val_dataset,
            test_dataset = None,
            batch_size=32,
            collate_fn=None,
            ):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        if test_dataset is not None:
            self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.collate_fn = collate_fn

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=2,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=2,
            persistent_workers=True
        )

    def test_dataloader(self):
        if not hasattr(self, 'test_dataset'):
            return None

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=2,
            persistent_workers=True
        )

# ---------------
# BatteryPipeline
# ---------------

class BatteryPipeline(pl.LightningModule):
    def __init__(
        self,
        model,
        denormalize_y=nn.Identity(),
        loss_type='mse',  # 'mse', 'huber', 'bce'
        huber_delta=1.0,  # Only for huber loss
        learning_rate=1e-3,
        metrics=None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model', 'denormalize_y'])
        self.model = model
        self.denormalize_y = denormalize_y

        # Configure loss function
        self.loss_fn = self._get_loss_fn(loss_type, huber_delta)
        
        # Configure metrics

        # Initialize metrics
        self._init_metrics(metrics)

    def _get_loss_fn(self, loss_type, huber_delta):
        loss_mapping = {
            'mse': nn.MSELoss(),
            'huber': nn.HuberLoss(delta=huber_delta),
            'bce': nn.BCELoss()
        }
        if loss_type not in loss_mapping:
            raise ValueError(f"Invalid loss_type: {loss_type}. Choose from {list(loss_mapping.keys())}")
        return loss_mapping[loss_type]

    def _init_metrics(self, metrics):
        metrics_set = {
            'mse': MeanSquaredError,
            'mae': MeanAbsoluteError,
            'mape': MeanAbsolutePercentageError,
            'r2': R2Score,
            'pcc': PearsonCorrCoef,
        }

        self.metric_classes = {}

        if isinstance(metrics, list):
            for key in metrics:
                if key in metrics_set.keys():
                    self.metric_classes[key] = metrics_set[key]
                else:
                    print(f'!WARNING: \'{key}\' metric is not supported')

        if metrics is None or metrics == 'all' or len(self.metric_classes) == 0:    
            self.metric_classes = metrics_set

        self.train_metrics = MetricCollection(
            {name: cls() for name, cls in self.metric_classes.items()},
            prefix='train_'
        )
        self.val_metrics = MetricCollection(
            {name: cls() for name, cls in self.metric_classes.items()},
            prefix='val_'
        )
        self.test_metrics = MetricCollection(
            {name: cls() for name, cls in self.metric_classes.items()},
            prefix='test_'
        )

    def forward(self, x, lengths):
        """Assumes model already has appropriate output activation"""
        return self.model(x, lengths)

    def _shared_step(self, batch):
        x = batch['x']
        lengths = batch['lengths']
        y_true = batch['y']
        y_pred = self(x, lengths)
        
        # Validate targets for BCE
        if isinstance(self.loss_fn, nn.BCELoss):
            y_true = y_true.float()
            if torch.any(y_true < 0) or torch.any(y_true > 1):
                raise ValueError(
                    f"Targets for BCE must be in [0,1] range. "
                    f"Found min={y_true.min():.4f}, max={y_true.max():.4f}"
                )
        
        loss = self.loss_fn(y_pred, y_true)
        return loss, y_pred, y_true

    def _update_metrics(self, metrics, y_pred, y_true):
        y_pred_denorm = self.denormalize_y(y_pred)
        y_true_denorm = self.denormalize_y(y_true)
        metrics(y_pred_denorm, y_true_denorm)

    def _log_metrics(self, metrics):
        results = metrics.compute()
        for name, value in results.items():
            self.log(name, value)
        metrics.reset()
        return results

    def training_step(self, batch, batch_idx):
        loss, y_pred, y_true = self._shared_step(batch)
        self._update_metrics(self.train_metrics, y_pred, y_true)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        self._log_metrics(self.train_metrics)

    def validation_step(self, batch, batch_idx):
        loss, y_pred, y_true = self._shared_step(batch)
        self._update_metrics(self.val_metrics, y_pred, y_true)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        self._log_metrics(self.val_metrics)

    def test_step(self, batch, batch_idx):
        loss, y_pred, y_true = self._shared_step(batch)
        self._update_metrics(self.test_metrics, y_pred, y_true)
        return {'loss': loss, 'y_pred': y_pred, 'y_true': y_true}

    def on_test_epoch_end(self):
        return self._log_metrics(self.test_metrics)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)