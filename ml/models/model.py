"""
Model architectures for fire detection.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict


class FireDetectionClassifier(nn.Module):
    """
    Fire detection classifier based on ResNet.
    
    Adapts ResNet for 4-channel Sentinel-2 input (B04, B08, B11, B12).
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        input_channels: int = 4,
        backbone: str = "resnet18",
        pretrained: bool = True,
        dropout: float = 0.3
    ):
        """
        Initialize classifier.
        
        Args:
            num_classes: Number of output classes
            input_channels: Number of input channels (4 for S2 bands)
            backbone: Backbone architecture (resnet18, resnet34, resnet50)
            pretrained: Whether to use pretrained weights
            dropout: Dropout rate
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.input_channels = input_channels
        
        # Load backbone
        if backbone == "resnet18":
            self.backbone = models.resnet18(pretrained=pretrained)
        elif backbone == "resnet34":
            self.backbone = models.resnet34(pretrained=pretrained)
        elif backbone == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Adapt first conv layer for 4 channels
        if input_channels != 3:
            original_conv = self.backbone.conv1
            self.backbone.conv1 = nn.Conv2d(
                input_channels,
                original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=False
            )
            
            # Initialize new conv layer
            if pretrained:
                # Copy weights from pretrained RGB conv
                # Average RGB weights for each Sentinel-2 channel
                with torch.no_grad():
                    # Initialize with average of RGB weights
                    self.backbone.conv1.weight[:, :3, :, :] = original_conv.weight
                    if input_channels > 3:
                        # Initialize extra channels with average
                        avg_weight = original_conv.weight.mean(dim=1, keepdim=True)
                        for i in range(3, input_channels):
                            self.backbone.conv1.weight[:, i:i+1, :, :] = avg_weight
        
        # Replace classifier head
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, num_classes)
        )
    
    def forward(self, x):
        """Forward pass."""
        return self.backbone(x)


def create_model(config: Dict) -> nn.Module:
    """
    Create model from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Model instance
    """
    model_config = config['model']
    
    model = FireDetectionClassifier(
        num_classes=model_config['num_classes'],
        input_channels=model_config['input_channels'],
        backbone=model_config['backbone'],
        pretrained=model_config['pretrained'],
        dropout=model_config['dropout']
    )
    
    return model


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model creation
    import yaml
    
    with open("ml/configs/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    print("Creating model...")
    model = create_model(config)
    
    print(f"\nModel: {config['model']['backbone']}")
    print(f"Input channels: {config['model']['input_channels']}")
    print(f"Output classes: {config['model']['num_classes']}")
    print(f"Parameters: {count_parameters(model):,}")
    
    # Test forward pass
    batch_size = 4
    dummy_input = torch.randn(batch_size, 4, 256, 256)
    
    print(f"\nTest forward pass:")
    print(f"Input shape: {dummy_input.shape}")
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Output shape: {output.shape}")
    print(f"Output (logits): {output[0]}")
