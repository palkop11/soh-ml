from glob import glob
import re
import bisect

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

# ----------------------
# Single Battery Dataset
# ----------------------

class SingleBatteryDataset(Dataset):
    """Represents all timeseries and corresponding targets of one battery

    ```
    Attributes
    ----------
    battery_path : str
        path to folder where all data for specified battery is stored

    features_step : int, default 1
        slicing step for extraction of features.

    features_npz_key : str, default 'current_voltage_unitless'
        key for acessing time series when reading .npz file

    normalize_features : callable, default None
        normalization transformation of features, by default keeps features
        timeseries unchanged

    normalize_targets : callable, default None
        normalization transformation of target, by default keeps target(s)
        unchanged

    n_diff : int or None, default None
        if not None, is used to make differencing:
        np.diff(features, axis = 0, n = n_diff)

    data : DataFrame
        stores info about cycle_numbers, paths and norm_cap (target) value

    Methods
    -------
    _prepare_data():
        used for construction of dataframe which stored in data attribute

    __getitem__(idx):
        returns features and target for the data.iloc[idx]
    """
    def __init__(
                self,
                battery_path: str,
                x_step : int = 1,
                x_npz_key : str = 'current_voltage_unitless',
                normalize : dict[callable] = {},
                n_diff : int | None = None,
            ):
        super().__init__()
        self.battery_path = battery_path
        self.x_step = x_step
        self.x_npz_key = x_npz_key

        self.normalize = {'x': None, 'y': None}
        self.normalize.update(normalize)

        self.n_diff = n_diff

        self.data = self._prepare_data()

    def _prepare_data(self):
        # Get time series paths
        ts_paths = glob(self.battery_path + '/time_series/*.npz')
        cycle_numbers = [int(re.findall(r'_(\d+)\.npz', p)[0]) for p in ts_paths]
        ts_df = pd.DataFrame({
            'cycle': cycle_numbers,
            'path': ts_paths
        }).sort_values('cycle')

        # Get targets
        targets_path = glob(self.battery_path + '/*targets.csv')[0]
        targets_df = pd.read_csv(targets_path).sort_values('number_cycle')

        # Merge and validate
        merged = pd.merge(
            targets_df,
            ts_df,
            left_on='number_cycle',
            right_on='cycle'
        ).drop(columns='cycle')

        assert len(merged) == len(ts_df), "Mismatch between cycles and targets"
        return merged

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Load time series data
        row = self.data.iloc[idx]
        x = np.load(row['path'])[self.x_npz_key][::self.x_step]
        if self.n_diff is not None:
            x = np.diff(x, axis = 0, n = self.n_diff)

        # Convert to tensors, normalize
        x = torch.tensor(x)  # [seq_len, 2]
        if self.normalize['x'] is not None:
            x = self.normalize['x'](x)

        # 'norm_cap' means that capacity is normalized to capacity at zero cycle,
        # but 'norm_cap' is not scaled by mean and std of dataset
        y = torch.tensor(row['norm_cap'], dtype=torch.float32)  # scalar
        if self.normalize['y'] is not None:
            y = self.normalize['y'](y)

        #y = y.reshape(-1, 1)
        return {
            'x': x,
            'y': y,
        }
    
# -----------------
# Composite Dataset
# -----------------

class CompositeBatteryDataset(Dataset):
    """Consits of one or more single battery datasets, have continous numbering
    of all battery cycles

    ```
    Attributes
    ----------
    batteries : list[SingleBatteryDataSet]
        list of instances of SingleBatteryDataset, which will be used for
        creation of composite dataset

    normalize_features : callable or None, default None
        Normalization transformation of features. If not None, used to set
        normalize_features attriubte of SingleBatteryDataset

    normalize_targets : callable or None, default None
        normalization transformation of target. If not None, used to set
        normalize_features attriubte of SingleBatteryDataset

    cumulative_sizes : list(int)
        List of cumulative sizes (or cumulative number of cycles) in each
        battery.
        cum_size(battery_{n}) = cum_size(battery_{n-1}) + len(battery_{n})

    Methods
    -------
    _set_normalize_transforms():
        if normalize transforms are not None, sets them for each single battery

    _compute_cumulative_sizes():
        computes and returns list of cumulative sizes of each battery, which is
        stored in cumulative_sizes attribute

    __getitem__(idx):
        finds which battery have cumulative size closest to (and greater than)
        given index using bisection. From found single battery_{n} cycle with the
        number = idx - cum_size(battery_{n-1}) is returned
    """
    def __init__(
            self,
            batteries : list[SingleBatteryDataset],
            normalize : dict[callable] = {},
        ):
        """
        Args:
            batteries: List of SingleBatteryDataset instances
            normalize_features : callable or None, default None
                normalization transform of features
            normalize_targets : callable or None, default None
                normalization transform of targets
        """
        self.batteries = batteries

        self.normalize = {'x': None, 'y': None}
        self.normalize.update(normalize)

        self._set_norm_for_single_batteries()
        self.cumulative_sizes = self._compute_cumulative_sizes()

    def _set_norm_for_single_batteries(self):
        for battery in self.batteries:
            battery.normalize.update(self.normalize)

    def _compute_cumulative_sizes(self):
        sizes = [len(b) for b in self.batteries]
        cumulative = []
        total = 0
        for size in sizes:
            total += size
            cumulative.append(total)
        return cumulative

    def __len__(self):
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0

    def __getitem__(self, idx):
        battery_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if battery_idx > 0:
            idx -= self.cumulative_sizes[battery_idx - 1]
        return self.batteries[battery_idx][idx]
    
# -------------------------------------
# Class for calculation of global stats
# -------------------------------------

class GlobalStatsCalculator:
    def __init__(self, stat_type: str | None, target: str = 'x'):
        """
        Calculate global statistics for normalization.

        Args:
            stat_type: Type of statistics to compute:
                - None : do not scale, just return bias = 0, and scale = 1,
                added for compatibility
                - 'minmax_zero_one': Scale to [0, 1] range
                - 'minmax_symmetric': Scale to [-1, 1] range
                - 'meanimax': Mean + max scaling
                - 'meanstd': Mean and standard deviation
            target: Data target to compute stats for ('x' or 'y')
        """
        self.stat_type = stat_type
        self.target = target
        self._validate_parameters()

    def _validate_parameters(self):
        valid_types = [None, 'minmax_zero_one', 'minmax_symmetric', 'meanimax', 'meanstd']
        valid_targets = ['x', 'y']
        if self.stat_type not in valid_types:
            raise ValueError(f"Invalid stat_type: {self.stat_type}. Choose from {valid_types}")
        if self.target not in valid_targets:
            raise ValueError(f"Invalid target: {self.target}. Choose from {valid_targets}")

    def compute(self, dataset):
        """Compute statistics, normalization bias, and scale."""
        if len(dataset) == 0:
            # stats, bias, scale
            return None, None, None

        if self.stat_type is None:
            # stats, bias, scale
            return None, 0.0, 1.0

        if self.target == 'y':
            return self._compute_y_stats(dataset)
        return self._compute_x_stats(dataset)

    def _compute_x_stats(self, dataset):
        """Compute statistics for time series features (x)"""
        device = dataset[0]['x'].device
        stats = {}
        eps = 1e-8  # Numerical stability

        # Initialize accumulators based on stat type
        if self.stat_type in ['minmax_zero_one', 'minmax_symmetric', 'meanimax']:
            global_min = torch.full((2,), float('inf'), device=device)
            global_max = torch.full((2,), float('-inf'), device=device)

        if self.stat_type in ['meanimax', 'meanstd']:
            total_sum = torch.zeros(2, device=device)
            total_count = 0

        if self.stat_type == 'meanstd':
            total_sum_sq = torch.zeros(2, device=device)

        # Process samples
        for sample in dataset:
            x = sample['x']
            if x.shape[0] == 0:
                continue

            # Track min/max for relevant stat types
            if self.stat_type in ['minmax_zero_one', 'minmax_symmetric', 'meanimax']:
                x_min = torch.min(x, dim=0).values
                x_max = torch.max(x, dim=0).values
                global_min = torch.minimum(global_min, x_min)
                global_max = torch.maximum(global_max, x_max)

            # Track sum/count for mean-based statistics
            if self.stat_type in ['meanimax', 'meanstd']:
                total_sum += torch.sum(x, dim=0)
                total_count += x.shape[0]

            # Track sum of squares for std calculation
            if self.stat_type == 'meanstd':
                total_sum_sq += torch.sum(x ** 2, dim=0)

        # Calculate final statistics
        if self.stat_type.startswith('minmax'):
            stats = {'min': global_min, 'max': global_max}

            if self.stat_type == 'minmax_zero_one':
                bias = global_min
                scale = global_max - global_min + eps
            else:  # minmax_symmetric
                midpoint = (global_min + global_max) / 2
                half_range = (global_max - global_min) / 2 + eps
                bias = midpoint
                scale = half_range

        elif self.stat_type == 'meanimax':
            global_mean = total_sum / total_count if total_count > 0 else torch.zeros(2, device=device)
            stats = {'mean': global_mean, 'min': global_min, 'max': global_max}
            bias = global_mean
            scale = torch.maximum(global_mean.abs(), global_max) + eps

        elif self.stat_type == 'meanstd':
            global_mean = total_sum / total_count if total_count > 0 else torch.zeros(2, device=device)
            global_var = (total_sum_sq / total_count) - (global_mean ** 2) + eps
            global_std = torch.sqrt(global_var)
            stats = {'mean': global_mean, 'std': global_std}
            bias = global_mean
            scale = global_std

        return stats, bias, scale

    def _compute_y_stats(self, dataset):
        """Compute statistics for scalar targets (y)"""
        y = torch.cat([sample['y'].unsqueeze(0) for sample in dataset])
        eps = 1e-8
        stats = {}

        if self.stat_type.startswith('minmax'):
            y_min = y.min()
            y_max = y.max()
            stats = {'min': y_min, 'max': y_max}

            if self.stat_type == 'minmax_zero_one':
                bias = y_min
                scale = y_max - y_min + eps
            else:  # minmax_symmetric
                midpoint = (y_min + y_max) / 2
                half_range = (y_max - y_min) / 2 + eps
                bias = midpoint
                scale = half_range

        elif self.stat_type == 'meanstd':
            stats = {'mean': y.mean(), 'std': y.std()}
            bias = stats['mean']
            scale = stats['std'] + eps

        return stats, bias, scale

# ------------------------------------
# Class for creation Composite dataset
# ------------------------------------

class DataSetCreation:
    def __init__(
            self,
            info,
            normalize : dict[callable] = {},
            fit_normalization : bool = False,
            normalization_types : dict[str] = {'x':'minmax', 'y':'meanstd'},
            StatsCalculator = GlobalStatsCalculator,
            x_step : int = 1,
            n_diff : int | None = None,
            Single : SingleBatteryDataset = SingleBatteryDataset,
            Composite : CompositeBatteryDataset = CompositeBatteryDataset,
        ):
        self.info = info
        self.Single = Single
        self.Composite = Composite

        self.normalize = {'x': None, 'y': None}
        self.normalize.update(normalize)

        self.fit_normalization = fit_normalization
        self.normalization_types = normalization_types
        self.StatsCalculator = StatsCalculator
        self.x_step = x_step
        self.n_diff = n_diff

        self.denormalize = {'x': None, 'y': None}
        self.dataset = None
        self.stats = {}

        self._check_normalization_types()
        self._create_main_dataset()

    def _check_normalization_types(self):
        if not self.fit_normalization:
            return None

        if isinstance(self.normalization_types, dict):
            for key, value in self.normalization_types.items():
                if (key not in ['x', 'y'] and \
                    value not in ['meanimax','minmax', 'meanstd']):
                    raise ValueError('wrong key/value for \'normalization_types\' \
dict argument')

        else:
            raise TypeError(f'type {type(self.normalization_types)} is not \
supported for \'normalization_types\' argument')

    def _create_composite_dataset(self):
        batteries = []

        for i, row in self.info.iterrows():
            batt = self.Single(
                    row['battery_path'],
                    x_step = self.x_step,
                    n_diff = self.n_diff,
                )
            batteries.append(batt)

        dataset = self.Composite(
                batteries,
                normalize = self.normalize,
            )
        return dataset

    def _set_stats_norm_denorm(self):
        dataset = self._create_composite_dataset()

        x_stats_calculator = self.StatsCalculator(
                stat_type = self.normalization_types['x'],
                target = 'x',
            )
        x_stats, x_bias, x_scale = x_stats_calculator.compute(dataset)

        y_stats_calculator = self.StatsCalculator(
                stat_type = self.normalization_types['y'],
                target = 'y',
            )
        y_stats, y_bias, y_scale = y_stats_calculator.compute(dataset)

        # Ensure tensors are on CPU for consistent behavior
        x_bias = x_bias.cpu() if torch.is_tensor(x_bias) else torch.tensor(x_bias)
        x_scale = x_scale.cpu() if torch.is_tensor(x_scale) else torch.tensor(x_scale)
        y_bias = y_bias.cpu() if torch.is_tensor(y_bias) else torch.tensor(y_bias)
        y_scale = y_scale.cpu() if torch.is_tensor(y_scale) else torch.tensor(y_scale)

        # Handle minmax_symmetric case properly
        if self.normalization_types['x'] == 'minmax_symmetric':
            self.normalize['x'] = lambda x: (x - x_bias) / x_scale
            self.denormalize['x'] = lambda x: x * x_scale + x_bias
        else:
            self.normalize['x'] = lambda x: (x - x_bias) / x_scale
            self.denormalize['x'] = lambda x: x * x_scale + x_bias

        if self.normalization_types['y'] == 'minmax_symmetric':
            self.normalize['y'] = lambda y: (y - y_bias) / y_scale
            self.denormalize['y'] = lambda y: y * y_scale + y_bias
        else:
            self.normalize['y'] = lambda y: (y - y_bias) / y_scale
            self.denormalize['y'] = lambda y: y * y_scale + y_bias

        self.stats['x'] = x_stats
        self.stats['y'] = y_stats

        # Test normalization round-trip
        test_x = torch.randn(10, 2)
        norm_x = self.normalize['x'](test_x)
        denorm_x = self.denormalize['x'](norm_x)
        assert torch.allclose(test_x, denorm_x, atol=1e-6), "X normalization round-trip failed"

        test_y = torch.randn(1)
        norm_y = self.normalize['y'](test_y)
        denorm_y = self.denormalize['y'](norm_y)
        assert torch.allclose(test_y, denorm_y, atol=1e-6), "Y normalization round-trip failed"

    def _create_main_dataset(self):
        if self.fit_normalization:
            self._set_stats_norm_denorm()
        self.dataset = self._create_composite_dataset()