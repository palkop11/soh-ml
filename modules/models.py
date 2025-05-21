# ----------
# LSTM model
# ----------

class LSTMModel(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=2, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )
        self.regressor = nn.Linear(hidden_size, output_size)

    def forward(self, x, lengths):
        # Pack padded sequence
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=True
        )
        output, (h_n, c_n) = self.lstm(packed)

        # Use last hidden state
        last_hidden = h_n[-1]
        return self.regressor(last_hidden).squeeze()

# --------
# CNN-LSTM
# --------

class CNNLSTMModel(nn.Module):
    def __init__(self,
                 input_size=2,
                 cnn_hidden=16,
                 lstm_hidden_size=64,
                 num_layers=2,
                 output_size=1,
                 dropout_prob=0.25,
                 output_activation = 'sigmoid'):
        super().__init__()

        # CNN block
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, 4, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(4, 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(8, cnn_hidden, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(cnn_hidden),
            nn.ReLU(),
        )

        # Dropout after CNN
        self.cnn_dropout = nn.Dropout(dropout_prob)

        # LSTM with inter-layer dropout
        self.lstm = nn.LSTM(
            input_size=cnn_hidden,
            hidden_size=lstm_hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout_prob if num_layers > 1 else 0  # Dropout between layers
        )

        # Final dropout before regressor
        self.final_dropout = nn.Dropout(dropout_prob)
        self.regressor = nn.Linear(lstm_hidden_size, output_size)
        self.output_activation = self._set_output_activation(output_activation)

    def _set_output_activation(self, output_activation):
        activations = {
            'tanh': nn.Tanh,
            'sigmoid': nn.Sigmoid,
            'relu': nn.ReLU,
        }

        if output_activation is None:
            return nn.Identity()

        if output_activation not in activations.keys():
            raise ValueError(f'activation {output_activation} wasnt found!')

        return activations[output_activation]()


    def forward(self, x, lengths):
        # Input shape: [batch_size, seq_len, 2]
        x = x.permute(0, 2, 1)  # [batch_size, 2, seq_len]

        # CNN processing
        x = self.cnn(x)
        x = self.cnn_dropout(x)

        # Calculate compressed lengths
        compressed_lengths = torch.div(lengths, 32, rounding_mode='floor').clamp(min=1)

        # Prepare for LSTM
        x = x.permute(0, 2, 1)  # [batch_size, compressed_seq_len, cnn_hidden]

        # Pack and process
        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            compressed_lengths.cpu(),
            batch_first=True,
            enforce_sorted=True
        )

        # LSTM processing
        output, (h_n, c_n) = self.lstm(packed)

        # Final dropout and regression
        last_hidden = self.final_dropout(h_n[-1])
        output = self.regressor(last_hidden).squeeze()
        output = self.output_activation(output)
        #return output
        return output.reshape(-1, 1)

# ------------------------
# CNN-LSTM for overfitting
# ------------------------

class CNN_LSTM_overfit_Model(nn.Module):
    """Not for actual learning, for testing if pipeline works correct"""
    def __init__(self,
                 input_size=2,
                 cnn_hidden=32,
                 lstm_hidden_size=32,
                 num_layers=1,
                 output_size=1,
                 dropout_prob=0.0,
                 regressor_hidden_dim = 1024,
                 output_activation = 'tanh',):
        super().__init__()

        # CNN block
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, 4, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(4, 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(8, cnn_hidden, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(cnn_hidden),
            nn.ReLU(),
        )

        # Dropout after CNN
        self.cnn_dropout = nn.Dropout(dropout_prob)

        # LSTM with inter-layer dropout
        self.lstm = nn.LSTM(
            input_size=cnn_hidden,
            hidden_size=lstm_hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout_prob if num_layers > 1 else 0  # Dropout between layers
        )

        # Final dropout before regressor
        self.final_dropout = nn.Dropout(dropout_prob)
        self.regressor1 = nn.Linear(lstm_hidden_size, regressor_hidden_dim)
        self.regressor_activation = nn.Tanh()
        self.regressor2 = nn.Linear(regressor_hidden_dim, output_size)
        self.output_activation = self._set_output_activation(output_activation)

    def _set_output_activation(self, output_activation):
        activations = {
            'tanh': nn.Tanh,
            'sigmoid': nn.Sigmoid,
            'relu': nn.ReLU,
        }

        if output_activation is None:
            return nn.Identity()

        if output_activation not in activations.keys():
            raise ValueError(f'activation {output_activation} wasnt found!')

        return activations[output_activation]()


    def forward(self, x, lengths):
        # Input shape: [batch_size, seq_len, 2]
        x = x.permute(0, 2, 1)  # [batch_size, 2, seq_len]

        # CNN processing
        x = self.cnn(x)
        x = self.cnn_dropout(x)

        # Calculate compressed lengths
        compressed_lengths = torch.div(lengths, 32, rounding_mode='floor').clamp(min=1)

        # Prepare for LSTM
        x = x.permute(0, 2, 1)  # [batch_size, compressed_seq_len, cnn_hidden]

        # Pack and process
        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            compressed_lengths.cpu(),
            batch_first=True,
            enforce_sorted=True
        )

        # LSTM processing
        output, (h_n, c_n) = self.lstm(packed)

        # Final dropout and regression
        last_hidden = self.final_dropout(h_n[-1])
        output = self.regressor1(last_hidden).squeeze()
        output = self.regressor_activation(output)
        output = self.regressor2(output).squeeze()
        output = self.output_activation(output)
        return output