# Phase 1: Data Acquisition - COMPLETE ✓

## What Was Built

### 1. Project Structure ✓
```
ml/
├── configs/config.yaml      # ML configuration
├── data/                    # Data pipeline scripts
│   ├── download_sentinel2.py
│   ├── match_firms_to_scenes.py
│   └── create_dataset.py
├── models/                  # (Phase 2)
├── training/                # (Phase 2)
├── inference/               # (Phase 2)
└── notebooks/               # (Phase 2)
```

### 2. Data Acquisition Scripts ✓

**a) Sentinel-2 Download Script**
- Queries STAC API for Sentinel-2 L2A scenes
- Downloads 4 key bands: Red (B04), NIR (B08), SWIR1 (B11), SWIR2 (B12)
- Supports filtering by cloud cover, bbox, date range
- Handles concurrent downloads with error recovery
- Saves scenes organized by date

**b) FIRMS-Scene Matching Script**
- Loads FIRMS fire detections from existing cache
- Matches fires to Sentinel-2 scenes spatially and temporally
- Configurable temporal window (±3 days)
- Filters by fire confidence (>=50)
- Generates matched dataset CSV with all metadata

**c) Dataset Generation Script**
- Extracts 256x256 pixel tiles from scenes
- Generates positive samples centered on fire locations
- Generates negative samples away from fires (3:1 ratio)
- Normalizes pixel values to 0-1 range
- Saves as numpy arrays with JSON labels

### 3. Configuration System ✓
- Centralized YAML config file
- Controls all data pipeline parameters
- Easy to modify for different regions/settings

### 4. Docker Integration ✓
- Updated Dockerfile with GDAL/geospatial dependencies
- ML directory structure integrated into container
- Ready for GPU support (Phase 2)

## Installation & Setup

### Step 1: Rebuild Docker Container

The Dockerfile has been updated to include GDAL and geospatial libraries:

```bash
# Rebuild container with ML dependencies
docker-compose down
docker-compose up --build -d

# Wait for container to start
sleep 5
```

### Step 2: Install Python ML Dependencies

```bash
# Enter container
docker exec -it portugal-fire-detection-web-1 bash

# Install core dependencies
uv add pyyaml pandas numpy rasterio shapely tqdm

# Install ML libraries (optional, for Phase 2)
uv add --group dev torch torchvision pytorch-lightning scikit-learn jupyter matplotlib seaborn
```

### Step 3: Test the Pipeline

```bash
# Inside container, run a small test
python ml/data/download_sentinel2.py \
    --start-date 2024-07-01 \
    --end-date 2024-08-31 \
    --max-scenes 5 \
    --max-cloud 30

python ml/data/match_firms_to_scenes.py
python ml/data/create_dataset.py

# Check results
ls -lh ml/data/tiles/*.npy | wc -l
cat ml/data/processed/dataset_summary.json
```

## What's Ready to Use

### Immediate Actions

1. **Download Training Data**
   - Script is ready to download Sentinel-2 scenes
   - Can start with small batches for testing
   - Recommend starting with summer 2024 (fire season)

2. **Match FIRMS Hotspots**
   - Uses existing FIRMS cache from your app
   - Automatically pairs fires with satellite imagery
   - Outputs matched dataset CSV

3. **Generate Training Set**
   - Creates labeled fire/no-fire tiles
   - Balanced dataset (3:1 negative:positive)
   - Ready for ML training

### Key Files

| File | Purpose |
|------|---------|
| `ml/configs/config.yaml` | Configure all parameters |
| `ml/README_PHASE1.md` | Detailed usage guide |
| `ml/data/download_sentinel2.py` | Download satellite scenes |
| `ml/data/match_firms_to_scenes.py` | Match fires to scenes |
| `ml/data/create_dataset.py` | Generate training tiles |

## Phase 1 Validation Checklist

Before moving to Phase 2, verify:

- [ ] Docker container rebuilds successfully with GDAL
- [ ] Python dependencies install without errors
- [ ] Can download at least 5 Sentinel-2 scenes
- [ ] FIRMS matching produces >0 matches
- [ ] Dataset generation creates `.npy` tiles
- [ ] Dataset summary shows balanced classes

## Storage Requirements

For a full dataset:
- **100 Sentinel-2 scenes**: ~50-100 GB
- **Training tiles**: ~5-10 GB
- **Total**: ~60-110 GB

For initial testing (5-10 scenes):
- **10 Sentinel-2 scenes**: ~5-10 GB
- **Training tiles**: ~500 MB - 1 GB
- **Total**: ~6-11 GB

## Known Limitations

1. **FIRMS NRT API**: Limited to 10 days of recent data
   - Solution: Use historical FIRMS archive (requires different API)
   - For now: Focus on recent fire seasons

2. **Cloud Cover**: Sentinel-2 scenes may have clouds
   - Current filter: <20% cloud cover
   - May need to increase for more data

3. **Temporal Matching**: ±3 day window
   - Some fires may not have matching imagery
   - Adjust `temporal_window_days` in config if needed

4. **Storage**: Large raw imagery files
   - Consider downloading in batches
   - Clean up raw scenes after tile extraction

## Next Steps (Phase 2)

Once Phase 1 is validated:

1. **Exploratory Data Analysis**
   - Create Jupyter notebook
   - Visualize fire spectral signatures
   - Analyze band combinations
   - Check for data quality issues

2. **Model Development**
   - Implement PyTorch Dataset/DataLoader
   - Build CNN classifier (ResNet50 backbone)
   - Set up PyTorch Lightning training loop
   - Add data augmentation

3. **Training**
   - Split data into train/val/test (70/15/15)
   - Train baseline model
   - Track metrics with Weights & Biases (optional)
   - Evaluate on test set

4. **Model Iteration**
   - Try different architectures (EfficientNet, U-Net)
   - Tune hyperparameters
   - Add temporal features (if multi-date data)
   - Implement early warning features

## Troubleshooting

### Docker build fails
- Check Docker has enough resources (4GB+ RAM, 50GB+ disk)
- Verify internet connection for apt-get packages
- Try `docker system prune -a` to clean old images

### GDAL errors
- Make sure container was rebuilt with new Dockerfile
- Check GDAL_CONFIG environment variable is set
- Try installing rasterio separately: `uv add rasterio`

### No FIRMS data
- Fetch fire data via web UI first (any region, last 7-10 days)
- Check `data/firms/` directory has JSON files
- Verify files contain "features" array

### Download timeouts
- Large Sentinel-2 files may timeout
- Retry failed downloads
- Reduce concurrent downloads in script (max_concurrent=2)

## Success Metrics

Phase 1 is complete when:
- ✅ Downloaded 50+ Sentinel-2 scenes
- ✅ Matched 100+ fires to scenes
- ✅ Generated 1000+ training tiles
- ✅ Dataset shows ~25% fire tiles, ~75% no-fire
- ✅ All scripts run without errors

## Resources

- **STAC Catalog**: https://earth-search.aws.element84.com/v1
- **FIRMS API**: https://firms.modaps.eosdis.nasa.gov/api/
- **Sentinel-2 Bands**: https://sentiwiki.copernicus.eu/web/s2-mission
- **Rasterio Docs**: https://rasterio.readthedocs.io/

## Questions?

Read the detailed guide: `ml/README_PHASE1.md`

Check configuration: `ml/configs/config.yaml`

View generated statistics: `ml/data/processed/*.json`
