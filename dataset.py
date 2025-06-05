import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import os, sys


class TemporalDataset(Dataset):
    def __init__(self,seq_len:int, dir_path:str, is_train:bool=True, train_months:list[int]=[1,2,3,4,5,6,7,8], val_months:list[int]=[9,10]):
        """
        Initialize dataset with temporal split based on months
        Args:
            dir_path: Path to directory containing CSV files
            is_train: If True, load training months, else load validation months
            train_months: List of months to use for training
            val_months: List of months to use for validation
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
                
                feature_cols = [col for col in current_df.columns if col not in ['date', 'target','day']]
                for col in feature_cols:
                    temp_df[f'{stock_name}_{col}'] = current_df[col]

                if self.data is None:
                    self.data = temp_df
                else:
                    # Merge on 'date' column. This assumes 'date' column is consistent across all CSVs
                    self.data = pd.merge(self.data, temp_df, on='date', how='inner')

        # Convert date and create month column
        self.data['date'] = pd.to_datetime(self.data['date'])
        self.data['month'] = self.data['date'].dt.month
        
        # Split based on months
        if is_train:
            self.data = self.data[self.data['month'].isin(train_months)]
        else:
            self.data = self.data[self.data['month'].isin(val_months)]
            
        # Separate features and targets
        self.data['day'] = self.data['date'].dt.dayofweek
        target_cols = [col for col in self.data.columns if '_target' in col]
        feature_cols = [col for col in self.data.columns if col not in ['date', 'month'] and col not in target_cols]

        self.features = self.data[feature_cols].values
        self.targets = self.data[target_cols].values
        
        # Normalize features
        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(self.features)
        #TODO: can we make the date range dynamic? currently 1 month
    def __len__(self):
        #TODO: take only the amount of different months (T)
        return len(self.features) // self.seq_len
    
    def __getitem__(self, idx):
        #TODO: store x as a tensor of shape [T,num of stocks,num_of_features]
        #TODO: make overlap a hyper parameter in the c'tor and return x,y accordginly
        x = torch.FloatTensor(self.features[idx * self.seq_len : ((idx+ 1) * self.seq_len),:])
        y = torch.FloatTensor(np.array(self.targets[idx * self.seq_len : ((idx+ 1) * self.seq_len)]))
        return x, y