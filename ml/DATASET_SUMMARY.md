# Dataset Summary Dashboard

## Overview

Web-based dashboard to visualize your local training dataset with false color Sentinel-2 imagery.

## Features

- **📡 Sentinel-2 Statistics**: Total scenes, date ranges, unique dates
- **🔥 FIRMS Fire Data**: Fire detection counts by region
- **🎯 Training Tiles**: Processed tile counts with fire/no-fire splits
- **✅ Training Readiness**: Status indicator for model training
- **🖼️ False Color Samples**: SWIR2-NIR-Red composites for fire detection visualization
- **📊 Split Distribution**: Train/val/test tile counts and ratios
- **🔄 Live Refresh**: Regenerate summary from latest data

## Usage

### 1. Generate Summary

Run the summary script to analyze your local data and create false color images:

```bash
# From project root
docker compose exec web python ml/data/summarize_dataset.py
```

This will:
- Scan `data/ml/sentinel2/` for Sentinel-2 scenes
- Analyze `data/firms/` for fire detections
- Check `ml/data/processed/` for training tiles
- Generate false color composites in `data/samples/`
- Save summary to `data/dataset_summary.json`

### 2. View Dashboard

Open in browser:
```
http://localhost:8000/static/dataset.html
```

Or access via API:
```bash
# Get full summary
curl http://localhost:8000/api/dataset/summary

# Get quick stats
curl http://localhost:8000/api/dataset/stats

# Refresh summary
curl -X POST http://localhost:8000/api/dataset/summary/refresh
```

## False Color Composite

The dashboard displays **SWIR2-NIR-Red** false color composites optimized for fire detection:

### Band Mapping
- **Red Channel**: SWIR2 (B12, 2190nm) → Active fires appear **bright red/white**
- **Green Channel**: NIR (B08, 842nm) → Healthy vegetation appears **green**
- **Blue Channel**: Red (B04, 665nm) → Water and bare soil visualization

### Why This Combination?
1. **SWIR2 is highly sensitive to fire** - thermal anomalies stand out
2. **NIR shows vegetation health** - differentiates burned vs. unburned areas
3. **Red provides context** - shows water bodies and bare ground
4. **Excellent contrast** - fires are unmistakable bright red/white spots

## Data Sources

### Sentinel-2 Scenes
- Location: `data/ml/sentinel2/`
- Structure: `YYYY-MM-DD/scene_id/Bxx.tif`
- Bands needed: B04, B08, B11, B12

### FIRMS Data
- Location: `data/firms/`
- Format: GeoJSON
- Contains: Fire hotspot locations and metadata

### Processed Tiles
- Location: `ml/data/processed/splits/`
- Structure: `train|val|test/fire|no_fire/*.npy`
- Format: NumPy arrays (256×256×4)

## Dashboard Sections

### 1. Overview Cards
- **Sentinel-2 Scenes**: Total count and date range
- **Fire Detections**: FIRMS hotspot count by region
- **Training Tiles**: Total tiles with fire ratio
- **Status**: Ready/Not Ready for training

### 2. Training Splits Table
Shows distribution across train/val/test:
- Fire tile count per split
- No-fire tile count per split
- Total tiles and fire ratio
- Progress bars for visual comparison

### 3. False Color Samples
Gallery of up to 5 Sentinel-2 scenes:
- False color composite images
- Scene date and ID
- Hover to enlarge
- Click to view details

## API Endpoints

### GET `/api/dataset/summary`
Returns complete dataset summary:
```json
{
  "generated_at": "2025-10-15T12:00:00",
  "sentinel2": {
    "total_scenes": 50,
    "date_range": {"start": "2024-06-01", "end": "2024-09-30"},
    "unique_dates": 15
  },
  "firms": {
    "total_fire_detections": 1573,
    "regions": [
      {"name": "portugal", "total_fires": 430, "files": 3},
      {"name": "california", "total_fires": 1143, "files": 2}
    ]
  },
  "processed": {
    "total_tiles": 5000,
    "fire_tiles": 1000,
    "no_fire_tiles": 4000,
    "splits": {
      "train": {"fire": 700, "no_fire": 2800},
      "val": {"fire": 150, "no_fire": 600},
      "test": {"fire": 150, "no_fire": 600}
    }
  },
  "samples": [
    {
      "scene_id": "S2A_...",
      "date": "2024-07-15",
      "image_url": "/static/samples/sample_0_....png"
    }
  ],
  "status": {
    "ready_for_training": true
  }
}
```

### POST `/api/dataset/summary/refresh`
Regenerates summary by rescanning all data directories.

### GET `/api/dataset/stats`
Quick stats without loading full summary:
```json
{
  "sentinel2_scenes": 50,
  "fire_detections": 1573,
  "training_tiles": 5000,
  "fire_tiles": 1000,
  "no_fire_tiles": 4000,
  "ready_for_training": true
}
```

## Example Output

Console output when running summary script:

```
Generating dataset summary...
  Analyzing Sentinel-2 scenes...
  Analyzing FIRMS data...
  Analyzing processed tiles...
  Generating false color samples...

✓ Summary saved to data/dataset_summary.json

================================================================================
Dataset Summary
================================================================================

📡 Sentinel-2 Scenes
  Total scenes: 50
  Date range: 2024-06-01 to 2024-09-30
  Unique dates: 15

🔥 FIRMS Fire Detections
  Total fire detections: 1,573
  Files: 5
    - portugal: 430 fires (3 files)
    - california: 1,143 fires (2 files)

🎯 Processed Training Data
  Total tiles: 5,000
  Fire tiles: 1,000 (20.0%)
  No-fire tiles: 4,000

  Splits:
    Train: 3,500 tiles (700 fire, 2,800 no-fire)
    Val: 750 tiles (150 fire, 600 no-fire)
    Test: 750 tiles (150 fire, 600 no-fire)

🖼️  Generated 5 false color sample images

✅ Status
  Sentinel-2 data: ✓
  FIRMS data: ✓
  Processed tiles: ✓
  Ready for training: ✓

================================================================================
```

## Troubleshooting

### No scenes found
```bash
# Check data directory
ls -la data/ml/sentinel2/

# Download scenes
docker compose exec web python ml/download_sentinel2_openeo.py
```

### No sample images generated
```bash
# Ensure rasterio is installed
docker compose exec web uv pip install rasterio

# Check band files exist
ls data/ml/sentinel2/*/*/B*.tif

# Regenerate with verbose output
docker compose exec web python ml/data/summarize_dataset.py --samples 5
```

### Images not showing in browser
```bash
# Check samples directory
ls -la data/samples/

# Verify static mount
docker compose logs web | grep samples

# Check file permissions
chmod -R 755 data/samples/
```

### Summary returns 0 for everything
- No data has been downloaded yet
- Run data acquisition scripts first:
  ```bash
  docker compose exec web python ml/download_sentinel2_openeo.py
  docker compose exec web python ml/data/fetch_historical_firms.py
  ```

## Integration with Training Pipeline

Use this dashboard to verify data before training:

1. **Check Sentinel-2 scenes** - Ensure sufficient coverage
2. **Verify FIRMS data** - Confirm fire detections available
3. **Review processed tiles** - Check split distribution
4. **Inspect false color samples** - Visual quality check
5. **Confirm training readiness** - Status must be "Ready"

Once ready:
```bash
docker compose exec web python ml/models/train.py
```

## Performance

- **Summary generation**: 10-30 seconds (depending on data size)
- **False color generation**: ~1 second per scene
- **Dashboard load time**: <1 second (cached data)
- **Refresh**: 10-30 seconds (full rescan)

## Future Enhancements

- [ ] Interactive scene selection
- [ ] Zoom/pan on false color images
- [ ] FIRMS overlay on satellite images
- [ ] Download statistics as CSV
- [ ] Scene filtering by date/region
- [ ] Batch false color generation
- [ ] NDVI/NBR calculation and visualization

---

**Access Dashboard**: http://localhost:8000/static/dataset.html
