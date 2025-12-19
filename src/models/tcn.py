import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, d_model: int, k: int, dilation: int, dropout: float = 0.1):
        """
        Args:
            d_model: Feature dimension
            k: Kernel size for convolution
            dilation: Dilation rate
            dropout: Dropout probability
        """

        super().__init__()
        # Symmetric padding to keep sequence length unchanged (non-causal)
        pad = (k - 1) * dilation // 2
        self.net = nn.Sequential(
            nn.Conv1d(d_model, d_model, k, padding=pad, dilation=dilation),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, k, padding=pad, dilation=dilation),
        )
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.ReLU(inplace=True)
    
    def forward(self, x):
        """
        Args:
            x: [B, T, C] input features
            
        Returns:
            out: [B, T, C] output features
        """
        y = x.transpose(1, 2)  # [B, C, T]
        y = self.net(y).transpose(1, 2)  # [B, T, C]
        # Residual connection with layer norm
        out = self.norm(x + y)
        return self.act(out)


class DilatedTCN(nn.Module):
    def __init__(
        self,
        d_model: int,
        layers: int,
        kernel_size: int,
        dilations=None,
        dropout: float = 0.1
    ):
        """
        Args:
            d_model: Feature dimension
            layers: Number of TCN layers
            kernel_size: Convolution kernel size (typically 3 or 5)
            dilations: List of dilation rates. If None, uses [1, 2, 4, 8, ...]
            dropout: Dropout probability
        """
        super().__init__()
        if dilations is None:
            # exponentially increasing dilations
            dilations = [2 ** i for i in range(layers)]
        
        blocks = [
            ResidualBlock(d_model, kernel_size, d, dropout)
            for d in dilations
        ]
        self.net = nn.Sequential(*blocks)
    
    def forward(self, x):
        """
        Args:
            x: [B, T, C] input video features
            
        Returns:
            out: [B, T, C] encoded temporal features
        """
        return self.net(x)