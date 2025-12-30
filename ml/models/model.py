"""
HuggingFace Transformer models for fire detection.

Easy model swapping - just change model_name in config.yaml!
"""

import logging
from typing import Dict

import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModelForImageClassification, AutoConfig

logger = logging.getLogger(__name__)


class HuggingFaceFireDetector(nn.Module):
    """
    Fire detection model using HuggingFace vision transformers.
    
    Easily swap models by changing model_name in config.
    """
    
    def __init__(
        self,
        model_name: str = "apple/mobilevit-small",
        num_classes: int = 2,
        input_channels: int = 4,
        pretrained: bool = True,
        freeze_backbone: bool = False
    ):
        """
        Initialize model.
        
        Args:
            model_name: HuggingFace model identifier
            num_classes: Number of output classes
            input_channels: Number of input channels (4 for Sentinel-2)
            pretrained: Whether to use pretrained weights
            freeze_backbone: Whether to freeze backbone weights
        """
        super().__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.input_channels = input_channels
        
        logger.info(f"Loading model: {model_name}")
        
        # Load config and model
        if pretrained:
            self.model = AutoModelForImageClassification.from_pretrained(
                model_name,
                num_labels=num_classes,
                ignore_mismatched_sizes=True  # Allow different num_classes
            )
        else:
            config = AutoConfig.from_pretrained(model_name)
            config.num_labels = num_classes
            self.model = AutoModelForImageClassification.from_config(config)
        
        # Adapt first layer for 4-channel input if needed
        if input_channels != 3:
            self._adapt_first_layer(input_channels)
        
        # Freeze backbone if requested
        if freeze_backbone:
            self._freeze_backbone()
            logger.info("Backbone frozen - only training classification head")
        
        logger.info(f"Model loaded: {self.count_parameters():,} parameters")
    
    def _adapt_first_layer(self, input_channels: int):
        """Adapt first convolutional layer for non-RGB input."""
        # Find the first conv layer (model-specific)
        first_conv = None
        
        # Try common patterns
        if hasattr(self.model, 'mobilevit'):
            # MobileViT
            first_conv = self.model.mobilevit.conv_stem.convolution
        elif hasattr(self.model, 'swin'):
            # Swin Transformer
            first_conv = self.model.swin.embeddings.patch_embeddings.projection
        elif hasattr(self.model, 'vit'):
            # ViT
            first_conv = self.model.vit.embeddings.patch_embeddings.projection
        elif hasattr(self.model, 'deit'):
            # DeiT
            first_conv = self.model.deit.embeddings.patch_embeddings.projection
        
        if first_conv is None:
            logger.warning(f"Could not find first conv layer for {self.model_name}")
            logger.warning("Model may not work correctly with 4 channels")
            return
        
        # Replace first conv layer
        old_weight = first_conv.weight.data
        old_out_channels = old_weight.shape[0]
        kernel_size = old_weight.shape[2:]
        stride = first_conv.stride
        padding = first_conv.padding
        
        new_conv = nn.Conv2d(
            input_channels,
            old_out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=(first_conv.bias is not None)
        )
        
        # Initialize new channels
        with torch.no_grad():
            # Copy RGB weights
            new_conv.weight[:, :3, :, :] = old_weight[:, :3, :, :]
            
            # Initialize extra channel (NIR) with average of RGB
            if input_channels > 3:
                avg_weight = old_weight.mean(dim=1, keepdim=True)
                new_conv.weight[:, 3:, :, :] = avg_weight.repeat(1, input_channels - 3, 1, 1)
            
            if first_conv.bias is not None:
                new_conv.bias.data = first_conv.bias.data
        
        # Replace in model
        if hasattr(self.model, 'mobilevit'):
            self.model.mobilevit.conv_stem.convolution = new_conv
        elif hasattr(self.model, 'swin'):
            self.model.swin.embeddings.patch_embeddings.projection = new_conv
        elif hasattr(self.model, 'vit'):
            self.model.vit.embeddings.patch_embeddings.projection = new_conv
        elif hasattr(self.model, 'deit'):
            self.model.deit.embeddings.patch_embeddings.projection = new_conv
        
        logger.info(f"Adapted first layer: 3 → {input_channels} channels")
    
    def _freeze_backbone(self):
        """Freeze all parameters except classification head."""
        for name, param in self.model.named_parameters():
            if 'classifier' not in name and 'head' not in name:
                param.requires_grad = False
    
    def forward(self, x):
        """Forward pass."""
        outputs = self.model(x)
        return outputs.logits
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(config: Dict) -> nn.Module:
    """
    Create model from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Model instance
    """
    model_config = config['model']
    
    model = HuggingFaceFireDetector(
        model_name=model_config['model_name'],
        num_classes=model_config['num_classes'],
        input_channels=model_config['input_channels'],
        pretrained=model_config['pretrained'],
        freeze_backbone=model_config.get('freeze_backbone', False)
    )
    
    return model


if __name__ == "__main__":
    # Test model creation
    import yaml
    
    with open("ml/configs/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    print("Creating model...")
    model = create_model(config)
    
    print(f"\nModel: {config['model']['model_name']}")
    print(f"Input channels: {config['model']['input_channels']}")
    print(f"Output classes: {config['model']['num_classes']}")
    print(f"Parameters: {model.count_parameters():,}")
    
    # Test forward pass
    batch_size = 2
    dummy_input = torch.randn(batch_size, 4, 256, 256)
    
    print(f"\nTest forward pass:")
    print(f"Input shape: {dummy_input.shape}")
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Output shape: {output.shape}")
    print(f"Output (logits): {output[0]}")
