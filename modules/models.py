import torch.nn as nn
import torch
from torchinfo import summary

class UnifiedBatteryModel(nn.Module):
    def __init__(self,
                 input_size=2,
                 cnn_hidden_dim=16,
                 cnn_channels=[4, 8, 16],
                 lstm_hidden_size=64,
                 num_layers=2,
                 output_size=1,
                 dropout_prob=0.25,
                 regressor_hidden_dim=None,
                 output_activation='sigmoid'):
        super().__init__()
        self.use_cnn = cnn_hidden_dim is not None and cnn_hidden_dim > 0
        self.cnn_channels = cnn_channels
        self.compress_factor = 1

        # CNN layers if enabled
        if self.use_cnn:
            cnn_layers = []
            in_channels = input_size
            for i, out_channels in enumerate(cnn_channels):
                cnn_layers.extend([
                    nn.Conv1d(in_channels, out_channels, kernel_size=5 if i == 0 else 3, stride=2, padding=2 if i == 0 else 1),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=2, stride=2)
                ])
                in_channels = out_channels
                self.compress_factor *= 4  # Each block (conv+pool) reduces length by 4x

            cnn_layers.append(
                nn.Conv1d(in_channels, cnn_hidden_dim, kernel_size=3, stride=2, padding=1)
            )
            cnn_layers.extend([
                nn.BatchNorm1d(cnn_hidden_dim),
                nn.ReLU(),
            ])
            self.compress_factor *= 2  # Final conv reduces by 2x
            self.cnn = nn.Sequential(*cnn_layers)
            self.cnn_dropout = nn.Dropout(dropout_prob)
            lstm_input_size = cnn_hidden_dim
        else:
            lstm_input_size = input_size

        # LSTM
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout_prob if num_layers > 1 else 0
        )

        self.final_dropout = nn.Dropout(dropout_prob)

        # Regressor
        if regressor_hidden_dim:
            self.regressor = nn.Sequential(
                nn.Linear(lstm_hidden_size, regressor_hidden_dim),
                nn.Tanh(),
                nn.Linear(regressor_hidden_dim, output_size)
            )
        else:
            self.regressor = nn.Linear(lstm_hidden_size, output_size)
        
        self.output_activation = self._get_activation(output_activation)

    def _get_activation(self, activation_name):
        if activation_name == 'tanh':
            return nn.Tanh()
        elif activation_name == 'sigmoid':
            return nn.Sigmoid()
        elif activation_name == 'relu':
            return nn.ReLU()
        else:
            return nn.Identity()

    def forward(self, x, lengths):
        if self.use_cnn:
            # Process through CNN
            x = x.permute(0, 2, 1)
            x = self.cnn(x)
            x = self.cnn_dropout(x)
            x = x.permute(0, 2, 1)
            compressed_lengths = torch.div(lengths, self.compress_factor, rounding_mode='floor').clamp(min=1)
        else:
            compressed_lengths = lengths
        
        # Pack and process
        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            compressed_lengths.cpu(),
            batch_first=True,
            enforce_sorted=True
        )

        # LSTM
        output, (h_n, c_n) = self.lstm(packed)

        # Regressor
        last_hidden = self.final_dropout(h_n[-1])
        output = self.regressor(last_hidden)
        output = self.output_activation(output)
        return output.reshape(-1, 1)
    
def make_model_summary(model, seq_lengths = [400, 16000]):
    for seq_length in seq_lengths:
        batch_size = 1
        input_size = 2
        dummy_x = torch.randn(batch_size, seq_length, input_size)
        dummy_lengths = torch.tensor([seq_length])  # Example: all sequences are full-length

        print('\ntensor shape for torchinfo model summary:', list(dummy_x.shape))

        summary(
            model,
            input_data={'x': dummy_x, 'lengths': dummy_lengths}
        )

        #print(output_summary, '\n')