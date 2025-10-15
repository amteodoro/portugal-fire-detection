# ML Fire Detection - Phase 1: Data Acquisition

This directory contains scripts and tools for collecting and preparing satellite imagery data for fire detection machine learning.

## Overview

Phase 1 implements the data acquisition pipeline:
1. **Download Sentinel-2 scenes** matching fire events
2. **Match FIRMS hotspots** to available imagery
3. **Generate training dataset** with labeled fire/no-fire tiles

## Directory Structure

```
ml/
├── configs/
│   └── config.yaml           # Main configuration file
├── data/
│   ├── raw/                  # Downloaded Sentinel-2 scenes
│   ├── processed/            # Matched dataset CSV
│   ├── tiles/                # Extracted image tiles (numpy arrays)
│   └── labels/               # Tile labels (JSON)
├── models/                   # Model architectures (Phase 2)
├── training/                 # Training scripts (Phase 2)
├── inference/                # Inference pipeline (Phase 3)
└── notebooks/                # Jupyter notebooks for EDA
```

## Workflow

### Step 1: Download Sentinel-2 Imagery

Download satellite scenes covering Portugal for the specified time period:

```bash
# Run inside Docker container
docker exec -it portugal-fire-detection-web-1 bash

# Download scenes (starts small for testing)
python ml/data/download_sentinel2.py \
    --start-date 2024-06-01 \
    --end-date 2024-09-30 \
    --max-scenes 10 \
    --max-cloud 20
```

**Arguments:**
- `--start-date`: Start date (YYYY-MM-DD)
- `--end-date`: End date (YYYY-MM-DD)
- `--max-scenes`: Limit number of scenes (for testing)
- `--max-cloud`: Maximum cloud cover percentage
- `--bbox`: Optional bbox override (lon_min lat_min lon_max lat_max)

**Output:**
- Raw Sentinel-2 scenes saved to `data/ml/sentinel2/`
- Each scene directory contains: `B04.tif`, `B08.tif`, `B11.tif`, `B12.tif`, `metadata.json`

### Step 2: Match FIRMS Fires to Scenes

Match fire detections from FIRMS with downloaded Sentinel-2 scenes:

```bash
python ml/data/match_firms_to_scenes.py
```

**What it does:**
- Reads FIRMS fire cache from `data/firms/`
- Scans downloaded Sentinel-2 scenes
- Matches fires to scenes spatially (bbox) and temporally (±3 days)
- Filters by fire confidence (>= 50)

**Output:**
- `ml/data/processed/matched_dataset.csv` - Fire-scene pairs
- `ml/data/processed/matching_statistics.json` - Summary statistics

**Matched Dataset Columns:**
- `fire_id`, `fire_lat`, `fire_lon`, `fire_date`, `fire_confidence`, `fire_frp`
- `scene_id`, `scene_date`, `scene_path`, `cloud_cover`, `date_diff_days`

### Step 3: Generate Training Tiles

Extract image tiles centered on fires (positive samples) and random locations (negative samples):

```bash
python ml/data/create_dataset.py
```

**What it does:**
- Reads matched dataset CSV
- For each scene with fires:
  - Extracts 256x256 pixel tiles around fire locations (positive samples)
  - Generates 3x negative samples per fire at random non-fire locations
- Normalizes pixel values to 0-1 range
- Saves tiles as `.npy` files with corresponding `.json` labels

**Output:**
- `ml/data/tiles/*.npy` - Image tiles (shape: 4 x 256 x 256)
- `ml/data/labels/*.json` - Tile metadata and labels
- `ml/data/processed/dataset_summary.json` - Dataset statistics

## Configuration

Edit `ml/configs/config.yaml` to customize:

### Key Parameters

```yaml
# Region and time range
region:
  bbox: [-9.5, 36.8, -6.0, 42.3]  # Portugal mainland
  
time_range:
  start_date: "2023-01-01"
  end_date: "2024-12-31"

# Sentinel-2 bands for fire detection
sentinel2:
  bands: ["B04", "B08", "B11", "B12"]  # Red, NIR, SWIR1, SWIR2
  max_cloud_cover: 20

# FIRMS matching
firms:
  temporal_window_days: 3  # Match scenes ±3 days from fire
  min_confidence: 50

# Tile extraction
tiles:
  size: 256  # Pixels
  negative_sample_ratio: 3  # 3 negative per 1 positive
```

## Data Requirements

### Estimated Storage

| Component | Size | Notes |
|-----------|------|-------|
| Sentinel-2 scenes (100 scenes) | ~50-100 GB | 4 bands per scene |
| Training tiles | ~5-10 GB | Depends on # of fires |
| **Total** | **~60-110 GB** | |

### Recommended Test Run

Start with a small dataset to validate the pipeline:

```bash
# 1. Download 5 scenes from summer 2024
python ml/data/download_sentinel2.py \
    --start-date 2024-07-01 \
    --end-date 2024-08-31 \
    --max-scenes 5

# 2. Match FIRMS (should find some fires in summer)
python ml/data/match_firms_to_scenes.py

# 3. Generate tiles
python ml/data/create_dataset.py
```

**Expected output:**
- ~5 Sentinel-2 scenes
- ~10-50 fire-scene matches (depending on fire activity)
- ~50-200 training tiles

## Troubleshooting

### No FIRMS data found
- Make sure you've fetched fire data via the web UI or API first
- FIRMS cache is located in `data/firms/`
- Try fetching for a wider date range (e.g., last 10 days)

### No Sentinel-2 scenes found
- Check cloud cover threshold (try `--max-cloud 50`)
- Verify date range overlaps with fire season (June-September)
- Check STAC API connectivity

### Download failures
- Large files may timeout - retry failed downloads
- Reduce `--max-scenes` for initial testing
- Check available disk space

## Next Steps (Phase 2)

Once Phase 1 is complete:

1. **Exploratory Data Analysis**
   - Create notebook to visualize fire signatures in different bands
   - Analyze class distribution and tile quality
   
2. **Model Development**
   - Implement CNN classifier (ResNet50 backbone)
   - Set up training pipeline with PyTorch Lightning
   - Add data augmentation

3. **Training**
   - Split dataset into train/val/test
   - Train initial baseline model
   - Evaluate and iterate

## Dependencies

Required Python packages (install via uv):
```bash
uv add pyyaml pandas numpy rasterio shapely tqdm
uv add --group ml torch torchvision pytorch-lightning scikit-learn
uv add --group ml jupyter matplotlib seaborn
```

## Quick Start Commands

```bash
# Enter container
docker exec -it portugal-fire-detection-web-1 bash

# Run complete Phase 1 pipeline
python ml/data/download_sentinel2.py --max-scenes 10
python ml/data/match_firms_to_scenes.py
python ml/data/create_dataset.py

# Check results
ls -lh ml/data/tiles/*.npy | wc -l  # Count tiles
cat ml/data/processed/dataset_summary.json  # View statistics
```

## Contact / Issues

For questions or issues with Phase 1 data acquisition, check:
- Configuration file: `ml/configs/config.yaml`
- Logs in console output
- Generated statistics: `ml/data/processed/*.json`
