import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import os, sys


class TemporalDataset(Dataset):
    def __init__(self, seq_len:int, dir_path:str, is_train:bool=True, window_span_for_volatile:int=5, 
                 train_ratio:float=0.8, random_seed:int=42):
        """
        Initialize dataset with random split
        Args:
            dir_path: Path to directory containing CSV files
            is_train: If True, load training split, else load validation split
            train_ratio: Ratio of data to use for training (0.8 = 80% train, 20% val)
            random_seed: Random seed for reproducible splits
        """
        dirs = os.listdir(dir_path)
        self.seq_len = seq_len
        # Initialize empty DataFrame with date column
        self.data = None

        for file in dirs:
            if file.endswith(".csv"):
                # Read current CSV
                current_df = pd.read_csv(os.path.join(dir_path, file))
                stock_name = file.replace('.csv', '')

                # Create a temporary DataFrame for this stock's data with prefixed columns
                temp_df = pd.DataFrame()
                temp_df['date'] = current_df['date']
                temp_df[f'{stock_name}_target'] = current_df['target'] - 1
                temp_df[f'{stock_name}_volatile'] = temp_df[f'{stock_name}_target'].rolling(window=window_span_for_volatile, min_periods=1).std()

                feature_cols = [col for col in current_df.columns if col not in ['date', 'target','day']]
                for col in feature_cols:
                    temp_df[f'{stock_name}_{col}'] = current_df[col]

                if self.data is None:
                    self.data = temp_df
                else:
                    # Merge on 'date' column. This assumes 'date' column is consistent across all CSVs
                    self.data = pd.merge(self.data, temp_df, on='date', how='inner')

        # Convert date and create day column
        self.data['date'] = pd.to_datetime(self.data['date'])
        self.data['day'] = self.data['date'].dt.dayofweek
        target_cols = [col for col in self.data.columns if '_target' in col]
        target_volatile_cols = [col for col in self.data.columns if '_volatile' in col]
        non_features_col = target_cols + target_volatile_cols
        feature_cols = [col for col in self.data.columns if col not in ['date', 'month'] and col not in non_features_col]

        # Ensure features, targets, and volatiles are not empty
        self.data = self.data.dropna(ignore_index=True)

        self.features = self.data[feature_cols].values
        self.targets = self.data[target_cols].values
        self.volatiles = self.data[target_volatile_cols].values

        # Normalize features
        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(self.features)
        
        # Create random split indices
        np.random.seed(random_seed)
        total_sequences = len(self.features) // self.seq_len
        indices = np.random.permutation(total_sequences)
        
        train_size = int(total_sequences * train_ratio)
        if is_train:
            self.indices = indices[:train_size]
        else:
            self.indices = indices[train_size:]
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # Use the random index to get the sequence
        sequence_idx = self.indices[idx]
        x = torch.FloatTensor(self.features[sequence_idx * self.seq_len : ((sequence_idx + 1) * self.seq_len),:])
        y = torch.FloatTensor(np.array(self.targets[sequence_idx * self.seq_len : ((sequence_idx + 1) * self.seq_len)]))
        vol = torch.FloatTensor(np.array(self.volatiles[sequence_idx * self.seq_len : ((sequence_idx + 1) * self.seq_len)]))
        return x, y, vol