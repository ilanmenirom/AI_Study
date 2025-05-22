import torch
import torch.nn as nn

class FCN(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 128, 256], num_classes=1):
        super(FCN, self).__init__()
        
        layers = []
        current_dim = input_dim
        
        # Create encoder layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Conv1d(current_dim, hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2)
            ])
            current_dim = hidden_dim
            
        # Create decoder layers
        for hidden_dim in reversed(hidden_dims[:-1]):
            layers.extend([
                nn.ConvTranspose1d(current_dim, hidden_dim, kernel_size=2, stride=2),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            ])
            current_dim = hidden_dim
            
        # Final layer
        layers.append(nn.Conv1d(current_dim, num_classes, kernel_size=1))
        
        self.fcn = nn.Sequential(*layers)
        
    def forward(self, x):
        # Reshape input to (batch_size, channels, sequence_length)
        x = x.unsqueeze(2)
        x = self.fcn(x)
        return x.squeeze(2)  # Remove the sequence dimension 