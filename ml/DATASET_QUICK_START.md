# Dataset Dashboard - Quick Start

## What It Does

Visualizes your local training dataset with:
- Sentinel-2 scene counts
- FIRMS fire detection statistics
- Training tile distribution (fire vs. no-fire)
- False color composites (SWIR2-NIR-Red) for fire visualization

## Quick Start

### 1. Generate Summary

```bash
docker compose exec web python ml/data/summarize_dataset.py
```

Output:
- Scans all data directories
- Creates false color PNG images in `data/samples/`
- Saves summary to `data/dataset_summary.json`

### 2. View Dashboard

Open browser: **http://localhost:8000/static/dataset.html**

You'll see:
- 📡 **Sentinel-2 Scenes**: How many satellite images you have
- 🔥 **Fire Detections**: FIRMS hotspot count
- 🎯 **Training Tiles**: Processed tiles ready for training
- ✅ **Status**: Whether you're ready to train
- 🖼️ **Samples**: False color images showing fires as bright red/white

### 3. Interpret False Color Images

**SWIR2-NIR-Red Composite** (optimized for fire detection):

- **Bright Red/White** = Active fires or hot spots
- **Green** = Healthy vegetation  
- **Brown/Yellow** = Burned areas or bare soil
- **Dark Blue** = Water bodies

## Example Scenarios

### Scenario 1: Just started, no data yet

**Dashboard shows:**
- 0 Sentinel-2 scenes
- 0 Fire detections
- 0 Training tiles
- Status: "Not Ready"

**Action:** Download data first
```bash
docker compose exec web python ml/download_sentinel2_openeo.py \
  --start-date 2024-06-01 \
  --end-date 2024-09-30
```

### Scenario 2: Have scenes and fires, need tiles

**Dashboard shows:**
- 50 Sentinel-2 scenes ✓
- 1,573 Fire detections ✓
- 0 Training tiles ✗
- Status: "Not Ready"

**Action:** Process tiles
```bash
docker compose exec web python ml/data/create_dataset.py
docker compose exec web python ml/models/prepare_splits.py
```

### Scenario 3: Ready to train!

**Dashboard shows:**
- 50 Sentinel-2 scenes ✓
- 1,573 Fire detections ✓
- 5,000 Training tiles ✓
  - Train: 3,500 (700 fire, 2,800 no-fire)
  - Val: 750 (150 fire, 600 no-fire)
  - Test: 750 (150 fire, 600 no-fire)
- Status: "Ready" ✓

**Action:** Start training!
```bash
docker compose exec web python ml/models/train.py --epochs 50
```

## API Access

### Get summary programmatically

```bash
# Full summary
curl http://localhost:8000/api/dataset/summary | jq

# Quick stats only
curl http://localhost:8000/api/dataset/stats | jq

# Refresh (rescans data)
curl -X POST http://localhost:8000/api/dataset/summary/refresh | jq
```

### Python usage

```python
import requests

# Get summary
response = requests.get('http://localhost:8000/api/dataset/summary')
data = response.json()

print(f"Scenes: {data['sentinel2']['total_scenes']}")
print(f"Fires: {data['firms']['total_fire_detections']}")
print(f"Training tiles: {data['processed']['total_tiles']}")
print(f"Ready: {data['status']['ready_for_training']}")
```

## Troubleshooting

**No images showing?**
```bash
# Make sure rasterio is installed
docker compose exec web uv pip install rasterio

# Check band files exist
docker compose exec web ls data/ml/sentinel2/*/*/B*.tif
```

**Summary returns all zeros?**
- You haven't downloaded data yet
- Run data acquisition scripts first

**Error loading dashboard?**
```bash
# Restart Docker
docker compose down && docker compose up -d

# Check logs
docker compose logs web | tail -50
```

## What to Look For

### Good Signs ✓
- Multiple Sentinel-2 scenes across different dates
- High fire detection count (>100 fires)
- Balanced fire/no-fire ratio (~20-30% fire tiles)
- Even train/val/test split distribution
- False color images show clear fires (bright red spots)

### Warning Signs ⚠️
- Too few scenes (<10) - need more data
- No fire detections - check FIRMS date range
- Extreme imbalance (<5% or >50% fire tiles)
- Missing bands in false color images
- All images look the same (no variation)

## Integration with Training

Before training, verify:

1. ✅ **Adequate data**: At least 20-30 Sentinel-2 scenes
2. ✅ **Fire diversity**: Multiple fire events across dates
3. ✅ **Balanced tiles**: 15-30% fire tiles
4. ✅ **All splits populated**: Train, val, test all have data
5. ✅ **Visual quality**: False color images show clear features

Then proceed:
```bash
docker compose exec web python ml/models/train.py
```

---

**Full documentation**: See `ml/DATASET_SUMMARY.md`
