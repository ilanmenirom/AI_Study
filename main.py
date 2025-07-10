import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import TemporalDataset
from model import FCN, create_fcn_model
from tqdm import tqdm
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from utils.sharpe_loss import SharpeLoss


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cuda'):
    model = model.to(device)
    best_val_loss = float('inf')
    
    # Lists to store losses for plotting
    train_losses_over_time = []
    val_losses_over_time = []
    
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
            
            # Add L1 regularization
            l1_lambda = 1e-5  # L1 regularization strength
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            loss = loss + l1_lambda * l1_norm
            
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
                
                # Add L1 regularization for validation loss too
                l1_lambda = 1e-5
                l1_norm = sum(p.abs().sum() for p in model.parameters())
                loss = loss + l1_lambda * l1_norm
                
                val_losses.append(loss.item())
        
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        
        # Store losses for plotting
        train_losses_over_time.append(avg_train_loss)
        val_losses_over_time.append(avg_val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Average Training Loss: {avg_train_loss:.4f}')
        print(f'Average Validation Loss: {avg_val_loss:.4f}')
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print('Saved new best model!')
        
        print('-' * 50)
    
    # Plot the training and validation losses using plotly
    epochs = list(range(1, num_epochs + 1))
    
    fig = go.Figure()
    
    # Add training loss trace
    fig.add_trace(go.Scatter(
        x=epochs,
        y=train_losses_over_time,
        mode='lines+markers',
        name='Training Loss',
        line=dict(color='blue', width=3),
        marker=dict(size=6)
    ))
    
    # Add validation loss trace
    fig.add_trace(go.Scatter(
        x=epochs,
        y=val_losses_over_time,
        mode='lines+markers',
        name='Validation Loss',
        line=dict(color='red', width=3),
        marker=dict(size=6)
    ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text='Training and Validation Loss Over Time',
            font=dict(size=20, color='black'),
            x=0.5
        ),
        xaxis=dict(
            title=dict(text='Epoch', font=dict(size=14)),
            tickfont=dict(size=12),
            gridcolor='lightgray'
        ),
        yaxis=dict(
            title=dict(text='Loss', font=dict(size=14)),
            tickfont=dict(size=12),
            gridcolor='lightgray'
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode='x unified',
        legend=dict(
            font=dict(size=12),
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='gray',
            borderwidth=1
        ),
        width=1000,
        height=600
    )
    
    # Add grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    # Save as HTML for interactive viewing
    # fig.write_html('training_losses.html')
    
    # Show the plot
    fig.show(renderer="browser")
    
    return train_losses_over_time, val_losses_over_time

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    seq_len = 16
    # Initialize datasets
    train_dataset = TemporalDataset(dir_path=r"./Dataset", is_train=True, seq_len=seq_len, window_span_for_volatile=5)
    val_dataset = TemporalDataset(dir_path=r"./Dataset", is_train=False, seq_len=seq_len, window_span_for_volatile=5)
    
    # Create data loaders
    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Get input and output dimensions from data
    sample_x, sample_y, _ = next(iter(train_loader))
    input_dim, output_dim = sample_x.shape[-1], sample_y.shape[-1]

    # Define model configuration
    model_config = {
        'input_dim': input_dim,
        'hidden_dims': [4],
        'output_dim': output_dim,
        'dropout_rate': 0.4,
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
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    
    # Train the model
    train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=40, device=device)
    
    print(f"\nTraining completed!")
    print(f"Final training loss: {train_losses[-1]:.4f}")
    print(f"Final validation loss: {val_losses[-1]:.4f}")
    print(f"Interactive loss plot saved as 'training_losses.html'")

if __name__ == '__main__':
    main()
