"""
Training script for fire detection model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import yaml
import argparse
from datetime import datetime
from tqdm import tqdm
import json

from model import create_model, count_parameters
from dataset import create_dataloaders


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


def train_epoch(model, loader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': running_loss / (batch_idx + 1),
            'acc': 100. * correct / total
        })
    
    return running_loss / len(loader), 100. * correct / total


def validate(model, loader, criterion, device):
    """Validate model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in tqdm(loader, desc="Validating"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    return running_loss / len(loader), 100. * correct / total


def save_checkpoint(model, optimizer, epoch, val_acc, checkpoint_dir, is_best=False):
    """Save model checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc
    }
    
    # Save latest
    torch.save(checkpoint, checkpoint_dir / 'latest.pth')
    
    # Save best
    if is_best:
        torch.save(checkpoint, checkpoint_dir / 'best.pth')
        print(f"  💾 Saved best model (val_acc: {val_acc:.2f}%)")
    
    # Save epoch checkpoint
    torch.save(checkpoint, checkpoint_dir / f'epoch_{epoch}.pth')


def train(config, args):
    """Main training function."""
    print("=" * 80)
    print("Fire Detection Model Training")
    print("=" * 80)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Create directories
    checkpoint_dir = Path(config['logging']['checkpoint_dir'])
    log_dir = Path(config['logging']['log_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dataloaders
    print("\nLoading dataset...")
    train_loader, val_loader, test_loader = create_dataloaders(
        config,
        batch_size=args.batch_size or config['training']['batch_size']
    )
    
    # Create model
    print("\nCreating model...")
    model = create_model(config).to(device)
    print(f"Parameters: {count_parameters(model):,}")
    
    # Loss function
    if config['loss']['type'] == 'focal':
        criterion = FocalLoss(
            alpha=config['loss']['alpha'],
            gamma=config['loss']['gamma']
        )
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    lr = args.lr or config['training']['learning_rate']
    if config['training']['optimizer'] == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Scheduler
    if config['training']['scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['epochs']
        )
    else:
        scheduler = None
    
    # TensorBoard
    writer = SummaryWriter(log_dir / datetime.now().strftime('%Y%m%d_%H%M%S'))
    
    # Training loop
    print("\nStarting training...")
    print("=" * 80)
    
    best_val_acc = 0.0
    patience_counter = 0
    patience = config['training']['early_stopping']['patience']
    
    epochs = args.epochs or config['training']['epochs']
    
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        print("-" * 80)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f"\nResults:")
        print(f"  Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f}  Val Acc:   {val_acc:.2f}%")
        
        # TensorBoard logging
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        # Save checkpoint
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
        
        save_checkpoint(model, optimizer, epoch, val_acc, checkpoint_dir, is_best)
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n⚠️  Early stopping triggered (patience: {patience})")
            break
        
        # Learning rate scheduling
        if scheduler:
            scheduler.step()
    
    writer.close()
    
    print("\n" + "=" * 80)
    print("✓ Training complete!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print("=" * 80)
    
    # Save training summary
    summary = {
        'best_val_acc': best_val_acc,
        'epochs_trained': epoch,
        'config': config,
        'timestamp': datetime.now().isoformat()
    }
    with open(checkpoint_dir / 'training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Train fire detection model")
    parser.add_argument('--config', default='ml/configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--lr', type=float, help='Learning rate')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Train
    train(config, args)


if __name__ == "__main__":
    main()
