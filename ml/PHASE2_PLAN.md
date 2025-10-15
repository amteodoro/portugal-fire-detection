# Phase 2: Model Training Plan

## Status

**Phase 1 Complete ✅**
- Sentinel-2 download pipeline (STAC + OpenEO)
- FIRMS fire data acquisition
- Scene-fire matching logic
- Dataset generation scripts
- Data in place for Portugal and California

**Phase 2: Model Development** (Current)

## Objectives

Build a deep learning model to detect fires from Sentinel-2 imagery using the multi-source labeled dataset.

## Architecture Options

### 1. **U-Net (Recommended for Segmentation)**
- **Use case**: Pixel-level fire detection
- **Input**: 4-band imagery (B04, B08, B11, B12)
- **Output**: Binary mask (fire/no-fire)
- **Pros**: Standard for segmentation, well-tested
- **Cons**: Requires dense labels

### 2. **ResNet + Classifier (Tile Classification)**
- **Use case**: Tile-level fire presence
- **Input**: 256x256 tiles, 4 bands
- **Output**: Binary (contains fire: yes/no)
- **Pros**: Simpler, fewer labels needed
- **Cons**: No spatial localization

### 3. **Attention U-Net**
- **Use case**: Advanced segmentation with attention
- **Input**: Multi-band patches
- **Output**: Fire probability map
- **Pros**: Better feature focus, state-of-the-art
- **Cons**: More complex, longer training

## Recommended Approach

**Start with Option 2 (ResNet Classifier)**, then iterate to U-Net:

### Phase 2a: Binary Classifier
1. Train ResNet18/34 on tile classification
2. Use FIRMS points as positive labels (fire tiles)
3. Random tiles as negative (no-fire)
4. **Fast iteration**, baseline metrics

### Phase 2b: Segmentation Model
1. U-Net for pixel-level detection
2. Use FIRMS locations to create fire masks
3. More precise localization
4. **Production model**

## Dataset Structure

```
data/processed/
├── train/
│   ├── fire/          # Tiles containing fires
│   │   ├── tile_001.npy  # 256x256x4 array
│   │   ├── tile_002.npy
│   │   └── ...
│   └── no_fire/       # Background tiles
│       ├── tile_100.npy
│       └── ...
├── val/
│   ├── fire/
│   └── no_fire/
└── test/
    ├── fire/
    └── no_fire/
```

## Implementation Steps

### Step 1: Dataset Refinement
```bash
# Already have: data/processed/fire_dataset/
# Contains: scenes + FIRMS matches

# Create train/val/test splits
python ml/models/prepare_splits.py
```

### Step 2: Data Pipeline
```python
# PyTorch DataLoader
# Augmentations: rotation, flip, brightness
# Normalization for each band
```

### Step 3: Model Implementation
```python
# ResNet-based classifier
# 4-channel input (custom first layer)
# Binary cross-entropy loss
# Adam optimizer
```

### Step 4: Training Loop
```bash
python ml/models/train.py \
  --model resnet18 \
  --epochs 50 \
  --batch-size 32 \
  --lr 0.001
```

### Step 5: Evaluation
```python
# Metrics: Accuracy, Precision, Recall, F1, AUC
# Confusion matrix
# False positive/negative analysis
```

### Step 6: Inference Pipeline
```python
# Load model
# Process new Sentinel-2 scenes
# Generate fire probability maps
# Export to GeoJSON for web display
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Framework | PyTorch |
| Vision | torchvision |
| Geospatial | rasterio, geopandas |
| Training | PyTorch Lightning (optional) |
| Tracking | TensorBoard / Weights & Biases |
| Deployment | ONNX export for inference |

## Success Metrics

### Minimum Viable Model (MVM)
- **Accuracy**: >85% on validation
- **Recall**: >80% (catch most fires)
- **Precision**: >70% (limit false alarms)
- **Inference**: <5 seconds per scene

### Production Target
- **Accuracy**: >90%
- **Recall**: >90%
- **Precision**: >85%
- **Inference**: <2 seconds per scene

## Timeline

| Week | Milestone |
|------|-----------|
| 1 | Data splits, DataLoader, baseline model |
| 2 | Training loop, initial results |
| 3 | Hyperparameter tuning, augmentation |
| 4 | Evaluation, deployment prep |

## Next Immediate Actions

1. ✅ **DONE**: OpenEO downloader for reliable S2 data
2. **TODO**: Create `ml/models/` directory structure
3. **TODO**: Implement train/val/test split script
4. **TODO**: Build PyTorch DataLoader
5. **TODO**: Implement ResNet classifier
6. **TODO**: Training script with checkpoints
7. **TODO**: Evaluation and metrics script

## Files to Create

```
ml/models/
├── __init__.py
├── prepare_splits.py       # Split dataset
├── dataset.py              # PyTorch Dataset class
├── model.py                # Model architecture
├── train.py                # Training loop
├── evaluate.py             # Evaluation metrics
└── inference.py            # Production inference
```

## Configuration

Add to `ml/configs/config.yaml`:
```yaml
model:
  architecture: resnet18
  input_channels: 4
  num_classes: 2
  
training:
  batch_size: 32
  epochs: 50
  learning_rate: 0.001
  weight_decay: 0.0001
  
data:
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15
  tile_size: 256
  augmentation: true
```

## Resources

- **PyTorch Tutorials**: https://pytorch.org/tutorials/
- **U-Net Paper**: https://arxiv.org/abs/1505.04597
- **Satellite Image Segmentation**: https://github.com/satellite-image-deep-learning
- **Fire Detection Research**: Search for "wildfire detection deep learning sentinel-2"

---

**Ready to proceed with Phase 2?** 

Start with:
```bash
docker compose exec web python ml/models/prepare_splits.py
```
