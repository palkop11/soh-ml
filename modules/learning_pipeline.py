from torch.nn.utils.rnn import pad_sequence

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics import MeanSquaredError, MeanAbsolutePercentageError

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
            denormalize_y = lambda y: y,
            loss_fn=nn.MSELoss(),
            learning_rate=1e-3
            ):

        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model

        # denormalize y for metrics calculation
        self.denormalize_y = denormalize_y

        # Initialize metrics
        self.train_mse = MeanSquaredError()
        self.train_mape = MeanAbsolutePercentageError()
        self.val_mse = MeanSquaredError()
        self.val_mape = MeanAbsolutePercentageError()
        self.test_mse = MeanSquaredError()
        self.test_mape = MeanAbsolutePercentageError()

    def forward(self, x, lengths):
        return self.model(x, lengths)

    def _shared_step(self, batch):
        x = batch['x']
        lengths = batch['lengths']
        y_true = batch['y']
        y_pred = self(x, lengths)
        loss = self.hparams.loss_fn(y_pred, y_true)
        return loss, y_pred, y_true

    def training_step(self, batch, batch_idx):
        loss, y_pred, y_true = self._shared_step(batch)

        # Update metrics
        y_pred_denorm = self.denormalize_y(y_pred)
        y_true_denorm = self.denormalize_y(y_true)

        self.train_mse(y_pred_denorm, y_true_denorm)
        self.train_mape(y_pred_denorm, y_true_denorm)

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        self.log('train_mse', self.train_mse.compute(), prog_bar=True)
        self.log('train_mape', self.train_mape.compute())
        self.train_mse.reset()
        self.train_mape.reset()

    def validation_step(self, batch, batch_idx):
        loss, y_pred, y_true = self._shared_step(batch)

        # Update metrics
        y_pred_denorm = self.denormalize_y(y_pred)
        y_true_denorm = self.denormalize_y(y_true)

        self.val_mse(y_pred_denorm, y_true_denorm)
        self.val_mape(y_pred_denorm, y_true_denorm)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        self.log('val_mse', self.val_mse.compute(), prog_bar=True)
        self.log('val_mape', self.val_mape.compute())
        self.val_mse.reset()
        self.val_mape.reset()

    def test_step(self, batch, batch_idx):
        loss, y_pred, y_true = self._shared_step(batch)

        # Denormalize for metrics
        y_pred_denorm = self.denormalize_y(y_pred)
        y_true_denorm = self.denormalize_y(y_true)

        # Update test metrics
        self.test_mse(y_pred_denorm, y_true_denorm)
        self.test_mape(y_pred_denorm, y_true_denorm)

        # Optional: Store predictions for later analysis
        self.log('test_loss', loss, on_epoch=True, prog_bar=True)
        return {'test_loss': loss, 'y_pred': y_pred, 'y_true': y_true}

    def on_test_epoch_end(self):
        # Log aggregated test metrics
        self.log('test_mse', self.test_mse.compute(), prog_bar=True)
        self.log('test_mape', self.test_mape.compute())
        self.test_mse.reset()
        self.test_mape.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)