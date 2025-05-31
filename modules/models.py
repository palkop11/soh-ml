import torch.nn as nn
import torch
from torchinfo import summary

class CNN_LSTM_model(nn.Module):
    def __init__(self,
                 input_size=2,
                 #cnn_hidden_dim=16,  # Now used only if not part of cnn_channels
                 cnn_channels=[4, 8, 16],  # Last element can be final channel dim
                 cnn_kernel_sizes=None,
                 cnn_strides=None,
                 cnn_paddings=None,
                 cnn_use_maxpool=None,  # List of booleans for maxpool per block
                 lstm_hidden_size=64,
                 num_layers=2,
                 output_size=1,
                 dropout_prob=0.,
                 regressor_hidden_dim=1024,
                 output_activation='sigmoid'):
        super().__init__()
        self.use_cnn = len(cnn_channels) > 0  # CNN enabled if channels provided
        
        # CNN configuration
        self.cnn_channels = cnn_channels

        self.compress_factor = 1

        # Set default CNN parameters if not provided
        default_kernels = [5] + [3] * (len(cnn_channels) - 1)
        default_strides = [2] * len(cnn_channels)
        default_paddings = [2] + [1] * (len(cnn_channels) - 1)
        default_maxpool = [True] * len(cnn_channels)

        self.cnn_kernel_sizes = cnn_kernel_sizes or default_kernels
        self.cnn_strides = cnn_strides or default_strides
        self.cnn_paddings = cnn_paddings or default_paddings
        self.cnn_use_maxpool = cnn_use_maxpool or default_maxpool

        # Parameter safety checks
        assert len(self.cnn_kernel_sizes) == len(self.cnn_channels)
        assert len(self.cnn_strides) == len(self.cnn_channels)
        assert len(self.cnn_paddings) == len(self.cnn_channels)
        assert len(self.cnn_use_maxpool) == len(self.cnn_channels)

        # CNN layers
        if self.use_cnn:
            cnn_layers = []
            in_channels = input_size
            
            # Build each CNN block
            for i, out_channels in enumerate(self.cnn_channels):
                # Conv layer
                conv = nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=self.cnn_kernel_sizes[i],
                    stride=self.cnn_strides[i],
                    padding=self.cnn_paddings[i]
                )
                cnn_layers.append(conv)
                cnn_layers.append(nn.BatchNorm1d(out_channels))
                cnn_layers.append(nn.ReLU())
                
                # Update compression factor
                self.compress_factor *= self.cnn_strides[i]
                
                # Optional max pooling
                if self.cnn_use_maxpool[i]:
                    cnn_layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
                    self.compress_factor *= 2  # Max pool reduces by 2x
                
                in_channels = out_channels

            self.cnn = nn.Sequential(*cnn_layers)
            self.cnn_dropout = nn.Dropout(dropout_prob)
            lstm_input_size = self.cnn_channels[-1]  # Use last channel dimension
        else:
            lstm_input_size = input_size

        # LSTM remains unchanged
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout_prob if num_layers > 1 else 0
        )

        self.final_dropout = nn.Dropout(dropout_prob)

        # Regressor remains unchanged
        if regressor_hidden_dim:
            self.regressor = nn.Sequential(
                nn.Linear(lstm_hidden_size, regressor_hidden_dim),
                nn.Tanh(),
                nn.Linear(regressor_hidden_dim, output_size)
            )
        else:
            self.regressor = nn.Linear(lstm_hidden_size, output_size)
        
        self.output_activation = self._get_activation(output_activation)

    # _get_activation remains unchanged
    def _get_activation(self, activation_name):
        if activation_name == 'tanh':
            return nn.Tanh()
        elif activation_name == 'sigmoid':
            return nn.Sigmoid()
        elif activation_name == 'relu':
            return nn.ReLU()
        else:
            return nn.Identity()

    # forward pass remains mostly unchanged
    def forward(self, x, lengths):
        if self.use_cnn:
            # Process through CNN
            x = x.permute(0, 2, 1)
            x = self.cnn(x)
            x = self.cnn_dropout(x)
            x = x.permute(0, 2, 1)
            
            # PRECISE LENGTH CALCULATION
            compressed_lengths = lengths.clone()
            for i in range(len(self.cnn_channels)):
                # Conv layer length calculation
                compressed_lengths = (
                    compressed_lengths + 
                    2 * self.cnn_paddings[i] - 
                    (self.cnn_kernel_sizes[i] - 1) - 1
                ) // self.cnn_strides[i] + 1
                
                # MaxPool layer calculation
                if self.cnn_use_maxpool[i]:
                    compressed_lengths = (
                        compressed_lengths + 
                        2 * 0 -  # padding=0
                        (2 - 1) - 1  # kernel_size=2, dilation=1
                    ) // 2 + 1
            
            compressed_lengths = compressed_lengths.clamp(min=1)
        else:
            compressed_lengths = lengths
    
    # Pack and process (rest of code unchanged)
        
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
    
def make_model_summary(
        model, 
        #seq_lengths = [379, 16674],
        seq_lengths = [256],
        dtype = torch.float32,
    ):
    for seq_length in seq_lengths:
        batch_size = 1
        input_size = 2
        dummy_x = torch.randn(batch_size, seq_length, input_size, dtype=dtype)
        dummy_lengths = torch.tensor([seq_length])  # Example: all sequences are full-length

        print('\ntensor shape for torchinfo model summary:', list(dummy_x.shape))

        summary(
            model,
            input_data={'x': dummy_x, 'lengths': dummy_lengths}
        )

        #print(output_summary, '\n')