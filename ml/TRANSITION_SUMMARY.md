# Phase 1 → Phase 2 Transition Summary

## What Was Done

### Problem Solved
✅ **STAC API 502 Errors** - Replaced unreliable AWS Element84 STAC endpoint with direct **OpenEO Copernicus API**

### New Implementation

#### 1. OpenEO Downloader (`ml/download_sentinel2_openeo.py`)
- Direct connection to Copernicus infrastructure
- OAuth2 authentication (browser-based login)
- Downloads Sentinel-2 L2A bands: B04 (Red), B08 (NIR), B11 (SWIR1), B12 (SWIR2)
- Compatible with existing pipeline (same output format)
- **Avoids 502 errors** that plagued STAC API

#### 2. Updated Dependencies (`pyproject.toml`)
Added ML pipeline requirements:
- `openeo>=0.27.0` - Copernicus API client
- `pystac-client>=0.7.0` - STAC support (legacy)
- `pyyaml>=6.0` - Config parsing
- `tqdm>=4.66.0` - Progress bars
- `rasterio>=1.3.0` - GeoTIFF I/O
- `numpy>=1.24.0` - Array operations
- `geopandas>=0.14.0` - Spatial data
- `shapely>=2.0.0` - Geometry operations

#### 3. Docker Configuration
- Added `ml/` directory to Docker volumes
- ML scripts now accessible in container

#### 4. Documentation
- **`ml/OPENEO_SETUP.md`**: Complete setup and usage guide
- **`ml/PHASE2_PLAN.md`**: Model training roadmap

## Usage

### Download Sentinel-2 Data (New Method)

```bash
# Install dependencies (already in container)
docker compose exec web uv pip install openeo pyyaml

# Download data (requires Copernicus account)
docker compose exec web python ml/download_sentinel2_openeo.py \
  --start-date 2025-10-03 \
  --end-date 2025-10-05 \
  --bbox -9.5 36.8 -6.0 42.3 \
  --max-cloud 30
```

**First run**: Opens browser for Copernicus authentication
**Subsequent runs**: Uses cached OAuth token

### Create Copernicus Account
Free registration: https://dataspace.copernicus.eu/

## Architecture Comparison

| Feature | Old (STAC) | New (OpenEO) |
|---------|-----------|--------------|
| **Reliability** | ❌ 502 errors | ✅ Stable |
| **Auth** | None | OAuth2 |
| **Setup** | Simple | Account required |
| **Speed** | Fast | Moderate |
| **Processing** | Client-side | Server-side option |
| **Endpoint** | AWS Element84 | Direct Copernicus |

## Phase 1 Status: COMPLETE ✅

- [x] Sentinel-2 STAC downloader (legacy, unreliable)
- [x] Sentinel-2 OpenEO downloader (production-ready)
- [x] FIRMS fire data fetching (NRT + historical)
- [x] Scene-fire matching logic
- [x] Tile extraction and dataset generation
- [x] Data collected for Portugal & California

**Output**: `data/processed/fire_dataset/` with labeled satellite imagery

## Phase 2: Model Training (Next)

### Recommended Approach
**Start Simple → Iterate to Complex**

#### Phase 2a: Binary Classifier (Week 1-2)
- ResNet18/34 for tile-level classification
- Input: 256×256×4 tiles (B04, B08, B11, B12)
- Output: Fire present (yes/no)
- Fast baseline, quick metrics

#### Phase 2b: Segmentation Model (Week 3-4)
- U-Net for pixel-level detection
- Output: Fire probability map
- Production deployment

### Implementation Plan

```
Week 1: Data prep
  - Train/val/test splits
  - PyTorch DataLoader
  - Data augmentation

Week 2: Baseline model
  - ResNet classifier
  - Training loop
  - Initial metrics

Week 3: Optimization
  - Hyperparameter tuning
  - Regularization
  - Cross-validation

Week 4: Deployment
  - Model export (ONNX)
  - Inference pipeline
  - Web integration
```

### Tech Stack

- **Framework**: PyTorch
- **Vision**: torchvision (pretrained models)
- **Geospatial**: rasterio, geopandas
- **Tracking**: TensorBoard
- **Deployment**: ONNX for inference

### Success Metrics

**Minimum Viable Model**:
- Accuracy: >85%
- Recall: >80% (catch fires)
- Precision: >70% (limit false alarms)

**Production Target**:
- Accuracy: >90%
- Recall: >90%
- Precision: >85%
- Inference: <2 seconds/scene

## Next Steps

### Immediate (Today)
1. ✅ OpenEO downloader implemented
2. ✅ Dependencies added
3. ✅ Documentation complete

### This Week
1. **Download production data** (OpenEO, 3-6 months of fire season)
2. **Create model directory structure** (`ml/models/`)
3. **Implement dataset splitter** (`prepare_splits.py`)
4. **Build PyTorch DataLoader** (`dataset.py`)

### Next Week
5. **Implement ResNet classifier** (`model.py`)
6. **Training script** (`train.py`)
7. **Evaluation metrics** (`evaluate.py`)

## Commands Reference

```bash
# Phase 1: Data Acquisition
docker compose exec web python ml/download_sentinel2_openeo.py
docker compose exec web python ml/data/fetch_historical_firms.py
docker compose exec web python ml/data/match_firms_to_scenes.py
docker compose exec web python ml/data/create_dataset.py

# Phase 2: Model Training (TODO)
docker compose exec web python ml/models/prepare_splits.py
docker compose exec web python ml/models/train.py
docker compose exec web python ml/models/evaluate.py
docker compose exec web python ml/models/inference.py
```

## Files Added/Modified

```
Modified:
  docker-compose.yml      # Added ml/ volume
  pyproject.toml          # Added ML dependencies

Added:
  ml/download_sentinel2_openeo.py    # OpenEO downloader
  ml/OPENEO_SETUP.md                 # Setup guide
  ml/PHASE2_PLAN.md                  # Training plan
  ml/TRANSITION_SUMMARY.md           # This file
```

## Resources

- **OpenEO Docs**: https://openeo.org/
- **Copernicus Portal**: https://dataspace.copernicus.eu/
- **PyTorch Tutorials**: https://pytorch.org/tutorials/
- **Satellite ML**: https://github.com/satellite-image-deep-learning

---

## Summary

✅ **Phase 1 Complete**: Data acquisition pipeline working with reliable OpenEO API
🚀 **Phase 2 Ready**: All groundwork laid for model training
📊 **Data Available**: Portugal & California fire scenes with FIRMS labels

**The STAC 502 issue is resolved. Ready to train fire detection models!**

See `ml/PHASE2_PLAN.md` for detailed next steps.
