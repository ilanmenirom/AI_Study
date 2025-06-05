import torch
import torch.nn as nn


class FCN(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dims,
        output_dim,
        dropout_rate,
        activation,
        use_batch_norm=True,
        use_layer_norm=False
    ):
        """
        Configurable FCN (Fully Convolutional Network)
        
        Args:
            input_dim (int): Input dimension
            hidden_dims (list): List of hidden dimensions for each layer
            output_dim (int): Output dimension. If None, will be same as input_dim
            dropout_rate (float): Dropout rate between layers
            activation (str): Activation function ('relu', 'gelu', 'tanh', 'sigmoid')
            use_batch_norm (bool): Whether to use batch normalization
            use_layer_norm (bool): Whether to use layer normalization
        """
        super(FCN, self).__init__()
        
        # Set output dimension
        self.output_dim = output_dim if output_dim is not None else input_dim
        
        # Get activation function
        self.activation = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid()
        }.get(activation.lower(), nn.ReLU())
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        # Add hidden layers
        for hidden_dim in hidden_dims:
            # Add fully connected layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Add batch norm if specified
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Add layer norm if specified
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            
            # Add activation
            layers.append(self.activation)
            
            # Add dropout
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        # Add output layer
        layers.append(nn.Linear(prev_dim, self.output_dim))
        
        # Create sequential model
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, T, num_features]
        
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, T, num_stocks]
        """
        batch_size, T, _ = x.shape
        
        # Reshape input to (batch_size * T * num_stocks, input_dim)
        x = x.reshape(-1, x.shape[-1])
        
        # Pass through network
        x = self.network(x)
        
        # Reshape output back to (batch_size, T, num_stocks)
        x = x.reshape(batch_size, T, self.output_dim)
        
        return x

def create_fcn_model(config):
    """
    Helper function to create FCN model from config dictionary
    
    Args:
        config (dict): Configuration dictionary with model parameters
        Example:
        {
            'input_dim': 10,
            'hidden_dims': [64, 128, 64],
            'output_dim': None,
            'dropout_rate': 0.2,
            'activation': 'relu',
            'use_batch_norm': True,
            'use_layer_norm': False
        }
    """
    return FCN(**config) 