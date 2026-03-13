"""
Training script for fire detection model.

Trains a HuggingFace vision transformer on Sentinel-2 tiles.
"""

import logging
import os
from pathlib import Path
from datetime import datetime
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from tqdm import tqdm
import yaml

from ml.models.model import create_model
from ml.models.dataset import create_dataloaders

def _create_model_for_config(config: dict):
    """Create the right model based on config['model_type']."""
    model_type = config.get("model_type", "mobilevit")
    if model_type == "multimodal":
        from ml.models.multimodal_model import create_multimodal_model
        return create_multimodal_model(config)
    return create_model(config)


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


class FireDetectionTrainer:
    """Trainer for fire detection models."""
    
    def __init__(self, config: Dict):
        """
        Initialize trainer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Using device: {self.device}")
        
        # Mixed precision training
        self.use_amp = config['training'].get('mixed_precision', False)
        if self.use_amp and self.device.type == 'cuda':
            self.scaler = torch.cuda.amp.GradScaler()
            logger.info("Mixed precision training enabled")
        else:
            self.scaler = None
        
        # Create directories
        self.checkpoint_dir = Path(config['logging']['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = Path(config['logging']['log_dir'])
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create experiment directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_name = config['logging']['experiment_name']
        self.exp_dir = self.log_dir / f"{exp_name}_{timestamp}"
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard
        self.writer = SummaryWriter(self.exp_dir)
        
        # Save config
        with open(self.exp_dir / 'config.yaml', 'w') as f:
            yaml.dump(config, f)
        
        logger.info(f"Experiment directory: {self.exp_dir}")
    
    def train_epoch(self, model, dataloader, criterion, optimizer, epoch):
        """Train for one epoch."""
        model.train()
        
        running_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
        for batch_idx, (inputs, labels, metadata) in enumerate(pbar):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.use_amp and self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                # Standard training
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
            
            # Metrics
            running_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy()
            preds = outputs.argmax(dim=1).detach().cpu().numpy()
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            avg_loss = running_loss / (batch_idx + 1)
            pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
        
        # Compute metrics
        metrics = self._compute_metrics(all_labels, all_preds, all_probs)
        metrics['loss'] = running_loss / len(dataloader)
        
        return metrics
    
    def validate(self, model, dataloader, criterion, epoch):
        """Validate the model."""
        model.eval()
        
        running_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")
            for batch_idx, (inputs, labels, metadata) in enumerate(pbar):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Metrics
                running_loss += loss.item()
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                preds = outputs.argmax(dim=1).cpu().numpy()
                
                all_probs.extend(probs)
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
                
                # Update progress bar
                avg_loss = running_loss / (batch_idx + 1)
                pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
        
        # Compute metrics
        metrics = self._compute_metrics(all_labels, all_preds, all_probs)
        metrics['loss'] = running_loss / len(dataloader)
        
        return metrics
    
    def _compute_metrics(self, labels, preds, probs):
        """Compute evaluation metrics."""
        accuracy = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average='binary', zero_division=0
        )
        
        # ROC-AUC
        try:
            roc_auc = roc_auc_score(labels, probs)
        except ValueError:
            roc_auc = 0.0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc
        }
    
    def save_checkpoint(self, model, optimizer, epoch, metrics, is_best=False):
        """Save model checkpoint."""
        from datetime import datetime as _dt
        training_metadata = {
            "training_date": _dt.utcnow().isoformat() + "Z",
            "epoch": epoch,
            "model_type": self.config.get("model_type", "mobilevit"),
            "val_metrics": metrics,
            "data_config": {
                "processed_dir": self.config.get("data", {}).get("processed_dir"),
                "train_split": self.config.get("dataset", {}).get("train_split"),
                "val_split": self.config.get("dataset", {}).get("val_split"),
                "test_split": self.config.get("dataset", {}).get("test_split"),
            },
        }
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config,
            'training_metadata': training_metadata,
        }
        
        # Save latest checkpoint
        latest_path = self.checkpoint_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model with val F1: {metrics['f1']:.4f}")
    
    def train(self):
        """Main training loop."""
        logger.info("="*50)
        logger.info("Starting Training")
        logger.info("="*50)
        
        # Create model
        logger.info("Creating model...")
        model = create_model(self.config)
        model = model.to(self.device)
        
        logger.info(f"Model: {self.config['model']['model_name']}")
        logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
        logger.info(f"Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        # Create dataloaders
        logger.info("Loading data...")
        dataloaders = create_dataloaders(self.config)
        
        if 'train' not in dataloaders or 'val' not in dataloaders:
            raise ValueError("Train and validation splits required")
        
        train_loader = dataloaders['train']
        val_loader = dataloaders['val']
        
        # Loss function
        criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        training_config = self.config['training']
        optimizer = optim.AdamW(
            model.parameters(),
            lr=training_config['learning_rate'],
            weight_decay=0.01
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=training_config['epochs']
        )
        
        # Early stopping
        early_stopping_config = training_config.get('early_stopping', {})
        early_stopping = EarlyStopping(
            patience=early_stopping_config.get('patience', 10),
            min_delta=early_stopping_config.get('min_delta', 0.001)
        )
        
        # Training loop
        best_val_f1 = 0.0
        
        for epoch in range(1, training_config['epochs'] + 1):
            logger.info(f"\nEpoch {epoch}/{training_config['epochs']}")
            logger.info(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Train
            train_metrics = self.train_epoch(model, train_loader, criterion, optimizer, epoch)
            
            # Validate
            val_metrics = self.validate(model, val_loader, criterion, epoch)
            
            # Log metrics
            self._log_metrics(epoch, train_metrics, val_metrics)
            
            # Save checkpoint
            is_best = val_metrics['f1'] > best_val_f1
            if is_best:
                best_val_f1 = val_metrics['f1']
            
            self.save_checkpoint(model, optimizer, epoch, val_metrics, is_best=is_best)
            
            # Learning rate scheduler
            scheduler.step()
            
            # Early stopping
            early_stopping(val_metrics['loss'])
            if early_stopping.early_stop:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break
        
        logger.info("\n" + "="*50)
        logger.info("Training Complete!")
        logger.info(f"Best validation F1: {best_val_f1:.4f}")
        logger.info(f"Best model saved to: {self.checkpoint_dir / 'best_model.pth'}")
        logger.info("="*50)
        
        self.writer.close()
    
    def _log_metrics(self, epoch, train_metrics, val_metrics):
        """Log metrics to console and TensorBoard."""
        # Console
        logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, "
                   f"Acc: {train_metrics['accuracy']:.4f}, "
                   f"F1: {train_metrics['f1']:.4f}, "
                   f"AUC: {train_metrics['roc_auc']:.4f}")
        
        logger.info(f"Val   - Loss: {val_metrics['loss']:.4f}, "
                   f"Acc: {val_metrics['accuracy']:.4f}, "
                   f"F1: {val_metrics['f1']:.4f}, "
                   f"AUC: {val_metrics['roc_auc']:.4f}")
        
        # TensorBoard
        for metric_name, value in train_metrics.items():
            self.writer.add_scalar(f'train/{metric_name}', value, epoch)
        
        for metric_name, value in val_metrics.items():
            self.writer.add_scalar(f'val/{metric_name}', value, epoch)


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train fire detection model")
    parser.add_argument('--config', default='ml/configs/config.yaml', help='Config file path')
    parser.add_argument(
        '--model-type',
        choices=['mobilevit', 'multimodal'],
        default=None,
        help='Model type (overrides config model_type)',
    )
    parser.add_argument('--device', default=None, help='Device (cuda/cpu)')
    
    # Training overrides
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--mixed-precision', action='store_true', help='Enable mixed precision training')
    parser.add_argument('--checkpoint-dir', type=str, default=None, help='Checkpoint directory')
    parser.add_argument('--experiment-name', type=str, default=None, help='Experiment name')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Apply CLI overrides
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    if args.batch_size is not None:
        config['dataset']['batch_size'] = args.batch_size
    if args.lr is not None:
        config['training']['learning_rate'] = args.lr
    if args.seed is not None:
        config['training']['seed'] = args.seed
    if args.mixed_precision:
        config['training']['mixed_precision'] = True
    if args.checkpoint_dir is not None:
        config['logging']['checkpoint_dir'] = args.checkpoint_dir
    if args.experiment_name is not None:
        config['logging']['experiment_name'] = args.experiment_name
    if args.model_type is not None:
        config['model_type'] = args.model_type
        if args.model_type == 'multimodal' and args.config == 'ml/configs/config.yaml':
            # Auto-switch to multimodal config default
            import os
            if os.path.exists('ml/configs/multimodal_config.yaml'):
                with open('ml/configs/multimodal_config.yaml') as f2:
                    import yaml as _yaml
                    mm_cfg = _yaml.safe_load(f2)
                mm_cfg.update({k: config[k] for k in config if k in mm_cfg})
                config = mm_cfg
                config['model_type'] = 'multimodal'

    # Override device if specified
    if args.device:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device if args.device == 'cuda' else ''
    
    # Set seed for reproducibility
    if 'seed' in config['training']:
        seed = config['training']['seed']
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        logger.info(f"Set random seed: {seed}")
    
    # Create trainer and train
    trainer = FireDetectionTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
