"""
Evaluation script for fire detection model.
"""

import torch
import yaml
import argparse
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)
import numpy as np
from tqdm import tqdm

from model import create_model
from dataset import create_dataloaders


def evaluate_model(model, loader, device):
    """
    Evaluate model on dataset.
    
    Returns:
        predictions, targets, probabilities
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for data, target in tqdm(loader, desc="Evaluating"):
            data = data.to(device)
            output = model(data)
            
            # Get probabilities
            probs = torch.softmax(output, dim=1)
            
            # Get predictions
            _, preds = output.max(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of fire class
    
    return np.array(all_preds), np.array(all_targets), np.array(all_probs)


def compute_metrics(preds, targets, probs):
    """Compute evaluation metrics."""
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(targets, preds)
    metrics['precision'] = precision_score(targets, preds, zero_division=0)
    metrics['recall'] = recall_score(targets, preds, zero_division=0)
    metrics['f1'] = f1_score(targets, preds, zero_division=0)
    
    # ROC AUC
    try:
        metrics['roc_auc'] = roc_auc_score(targets, probs)
    except:
        metrics['roc_auc'] = 0.0
    
    # Confusion matrix
    cm = confusion_matrix(targets, preds)
    metrics['confusion_matrix'] = cm.tolist()
    
    # Per-class metrics
    tn, fp, fn, tp = cm.ravel()
    metrics['true_negatives'] = int(tn)
    metrics['false_positives'] = int(fp)
    metrics['false_negatives'] = int(fn)
    metrics['true_positives'] = int(tp)
    
    # Specificity
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return metrics


def print_metrics(metrics, split_name="Test"):
    """Print metrics in a formatted way."""
    print("\n" + "=" * 80)
    print(f"{split_name} Set Results")
    print("=" * 80)
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:   {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision:  {metrics['precision']:.4f}")
    print(f"  Recall:     {metrics['recall']:.4f}")
    print(f"  F1 Score:   {metrics['f1']:.4f}")
    print(f"  ROC AUC:    {metrics['roc_auc']:.4f}")
    print(f"  Specificity: {metrics['specificity']:.4f}")
    
    print(f"\nConfusion Matrix:")
    cm = metrics['confusion_matrix']
    print(f"                Predicted")
    print(f"              No-Fire  Fire")
    print(f"  Actual  No-Fire  {cm[0][0]:4d}  {cm[0][1]:4d}")
    print(f"          Fire     {cm[1][0]:4d}  {cm[1][1]:4d}")
    
    print(f"\nDetailed:")
    print(f"  True Negatives:  {metrics['true_negatives']}")
    print(f"  False Positives: {metrics['false_positives']}")
    print(f"  False Negatives: {metrics['false_negatives']}")
    print(f"  True Positives:  {metrics['true_positives']}")
    
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Evaluate fire detection model")
    parser.add_argument('--config', default='ml/configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--checkpoint', default='best.pth',
                       help='Checkpoint to load (best.pth, latest.pth, or path)')
    parser.add_argument('--split', default='test',
                       choices=['train', 'val', 'test'],
                       help='Dataset split to evaluate')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print("=" * 80)
    print("Fire Detection Model Evaluation")
    print("=" * 80)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Load model
    print("\nLoading model...")
    model = create_model(config).to(device)
    
    # Load checkpoint
    checkpoint_dir = Path(config['logging']['checkpoint_dir'])
    checkpoint_path = checkpoint_dir / args.checkpoint
    
    if not checkpoint_path.exists():
        print(f"✗ Checkpoint not found: {checkpoint_path}")
        print("Available checkpoints:")
        for ckpt in checkpoint_dir.glob('*.pth'):
            print(f"  - {ckpt.name}")
        return
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Load dataset
    print(f"\nLoading {args.split} dataset...")
    train_loader, val_loader, test_loader = create_dataloaders(config)
    
    if args.split == 'train':
        loader = train_loader
    elif args.split == 'val':
        loader = val_loader
    else:
        loader = test_loader
    
    # Evaluate
    print("\nEvaluating...")
    preds, targets, probs = evaluate_model(model, loader, device)
    
    # Compute metrics
    metrics = compute_metrics(preds, targets, probs)
    
    # Print results
    print_metrics(metrics, split_name=args.split.capitalize())
    
    # Save results
    import json
    results_path = checkpoint_dir / f'{args.split}_results.json'
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
