import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import TemporalDataset
from model import FCN, create_fcn_model
from tqdm import tqdm
import numpy as np

from utils.sharpe_loss import SharpeLoss


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cuda'):
    model = model.to(device)
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_losses = []
        
        for batch_x, batch_y, batch_v in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Training'):
            # batch_x shape is: [batch_size,T,num of stocks * num_of_features]
            # batch_y shape is: [batch_size,T,num of stocks]
            batch_x, batch_y, batch_v = batch_x.to(device), batch_y.to(device), batch_v.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)  # [batch_size,T,num of stocks]
            loss = criterion(outputs, batch_y, batch_v)
            ## Vector multiplication
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Validation phase
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch_x, batch_y, batch_v in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Validation'):
                batch_x, batch_y, batch_v = batch_x.to(device), batch_y.to(device), batch_v.to(device)
                
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y, batch_v)
                val_losses.append(loss.item())
        
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Average Training Loss: {avg_train_loss:.4f}')
        print(f'Average Validation Loss: {avg_val_loss:.4f}')
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print('Saved new best model!')
        
        print('-' * 50)

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    seq_len = 4
    # Initialize datasets
    train_dataset = TemporalDataset(dir_path=r"./Dataset", is_train=True,seq_len=seq_len,window_span_for_volatile=5)
    val_dataset = TemporalDataset(dir_path=r"./Dataset", is_train=False,seq_len=seq_len,window_span_for_volatile=5)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Get input and output dimensions from data
    sample_x, sample_y, _ = next(iter(train_loader))
    input_dim, output_dim = sample_x.shape[-1], sample_y.shape[-1]

    # Define model configuration
    model_config = {
        'input_dim': input_dim,
        'hidden_dims': [64],
        'output_dim': output_dim,
        'dropout_rate': 0.2,
        'activation': 'relu',
        'use_batch_norm': True,
        'use_layer_norm': False
    }
    
    # Initialize model
    model = create_fcn_model(model_config)
    print("Model architecture:")
    print(model)
    
    # Define loss function and optimizer
    criterion = SharpeLoss(
        target_volatility=None,
        cost_rate=None,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device=device)

if __name__ == '__main__':
    main()
