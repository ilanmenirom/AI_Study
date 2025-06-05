import torch
import torch.nn as nn

class SharpeLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(SharpeLoss, self).__init__()
        self.eps = eps
        
    def forward(self, weights, returns):
        # Ensure weights are non-negative and sum to 1 for each day
        # weights shape: [batch_size, T, num_stocks]
        # returns shape: [batch_size, T, num_stocks]
        weights = torch.softmax(weights, dim=-1)  # TODO: ensure this is not done in the model
        
        # Calculate portfolio returns for each timestep
        # Sum across stocks dimension: [batch_size, T]
        portfolio_returns = torch.sum(weights * returns, dim=-1)
        
        # Calculate mean and std of portfolio returns across time dimension
        mean_returns = torch.mean(portfolio_returns, dim=1)  # [batch_size]
        std_returns = torch.std(portfolio_returns, dim=1)  # [batch_size]
        
        # Add epsilon to std to avoid division by zero
        if self.eps > std_returns * 0.001:
            raise ValueError("Epsilon is too large, consider reducing it")

        std_returns = std_returns + self.eps
        
        # Calculate Sharpe ratio for each batch
        sharpe_ratio = mean_returns / std_returns  # [batch_size]
        
        # Return negative mean Sharpe ratio as we want to maximize it
        return -torch.mean(sharpe_ratio)