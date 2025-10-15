# OpenEO Setup Guide

## Overview

We've added an **OpenEO-based** Sentinel-2 downloader as a more reliable alternative to the STAC API, which was experiencing intermittent 502 errors.

## Why OpenEO?

- **Direct Copernicus access**: Connects directly to Copernicus infrastructure
- **More reliable**: Avoids AWS Element84 STAC API issues
- **Server-side processing**: Can compute indices (NDVI, NBR) on the server
- **OAuth2 authentication**: Secure, browser-based authentication

## Prerequisites

### 1. Create Copernicus Account

Sign up for free at: https://dataspace.copernicus.eu/

### 2. Install Dependencies

Dependencies are already added to `pyproject.toml`:
```bash
docker compose exec web uv pip install openeo pyyaml
```

## Usage

### Basic Usage

```bash
# Download Sentinel-2 data for Portugal (from config)
docker compose exec web python ml/data/download_sentinel2_openeo.py

# Custom parameters
docker compose exec web python ml/data/download_sentinel2_openeo.py \
  --start-date 2025-10-03 \
  --end-date 2025-10-05 \
  --bbox -9.5 36.8 -6.0 42.3 \
  --max-cloud 30
```

### Authentication

The first time you run the script:
1. It will open a browser window
2. Log in with your Copernicus credentials
3. Grant access to the OpenEO client
4. The token is cached for future use

## Comparison: STAC vs OpenEO

| Feature | STAC API | OpenEO API |
|---------|----------|------------|
| Reliability | ❌ 502 errors | ✅ Stable |
| Setup | Simple | Requires login |
| Speed | Fast | Moderate |
| Processing | Client-side | Server-side option |
| Authentication | None | OAuth2 |

## Integration with Existing Pipeline

The OpenEO downloader:
- Uses the same config file (`ml/configs/config.yaml`)
- Saves to the same directory structure (`data/raw/sentinel2/`)
- Outputs the same band files (B04.tif, B08.tif, B11.tif, B12.tif)
- Creates compatible metadata.json files

**No changes needed** to downstream scripts (matching, dataset creation).

## Troubleshooting

### Authentication Failed
```
✗ Authentication failed: ...
```
**Solution**: Ensure you have a valid Copernicus account and internet connection.

### Module Not Found
```
ModuleNotFoundError: No module named 'openeo'
```
**Solution**: 
```bash
docker compose exec web uv pip install openeo pyyaml
```

### Download Timeout
OpenEO processes data server-side, which can take longer for large areas.
**Solution**: Use smaller bounding boxes or shorter date ranges.

## Next Steps

Once Sentinel-2 data is downloaded:

1. **Match with FIRMS fires**:
   ```bash
   docker compose exec web python ml/data/match_firms_to_scenes.py
   ```

2. **Create training dataset**:
   ```bash
   docker compose exec web python ml/data/create_dataset.py
   ```

3. **Start Phase 2 (Model Training)**

## Migration Path

To migrate from STAC to OpenEO:

1. Keep existing STAC script for quick tests
2. Use OpenEO for production data collection
3. Both scripts work with the same config and output format
