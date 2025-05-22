import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import os, sys


class TemporalDataset(Dataset):
    def __init__(self, dir_path:str, is_train:bool=True, train_months:list[int]=[1,2,3,4,5,6,7,8], val_months:list[int]=[9,10]):
        """
        Initialize dataset with temporal split based on months
        Args:
            dir_path: Path to directory containing CSV files
            is_train: If True, load training months, else load validation months
            train_months: List of months to use for training
            val_months: List of months to use for validation
        """
        dirs = os.listdir(dir_path)
        
        # Initialize empty DataFrame with date column
        self.data = None
        
        for file in dirs:
            if file.endswith(".csv"):
                # Read current CSV
                current_df = pd.read_csv(os.path.join(dir_path, file))
                
                if self.data is None:
                    # For first file, initialize the DataFrame with date column
                    self.data = current_df[['date', 'target']]
                    # Add features with stock name prefix
                    stock_name = file.replace('.csv', '')
                    feature_cols = [col for col in current_df.columns if col not in ['date', 'target']]
                    for col in feature_cols:
                        self.data[f'{stock_name}_{col}'] = current_df[col]
                else:
                    # For subsequent files, only add features with stock name prefix
                    stock_name = file.replace('.csv', '')
                    feature_cols = [col for col in current_df.columns if col not in ['date', 'target']]
                    for col in feature_cols:
                        self.data[f'{stock_name}_{col}'] = current_df[col]
        
        # Convert date and create month column
        self.data['date'] = pd.to_datetime(self.data['date'])
        self.data['month'] = self.data['date'].dt.month
        
        # Split based on months
        if is_train:
            self.data = self.data[self.data['month'].isin(train_months)]
        else:
            self.data = self.data[self.data['month'].isin(val_months)]
            
        # Assuming your features are all columns except 'date', 'month' and target
        self.features = self.data.drop(['date', 'month', 'target'], axis=1).values
        self.targets = self.data['target'].values
        
        # Normalize features
        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(self.features)
        #TODO: can we make the date range dynamic? currently 1 month
    def __len__(self):
        #TODO: take only the amount of different months (T)
        return len(self.features)
    
    def __getitem__(self, idx):
        #TODO: store x as a tensor of shape [T,num of stocks,num_of_features]
        x = torch.FloatTensor(self.features[idx])
        y = torch.FloatTensor([self.targets[idx]])
        return x, y 