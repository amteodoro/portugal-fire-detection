#!/usr/bin/env python3
"""Debug script to test tile extraction."""

import pandas as pd
import rasterio
from rasterio.transform import rowcol
from rasterio.warp import transform as reproject_coords
from pathlib import Path
import numpy as np

# Load matched dataset
matched_df = pd.read_csv("ml/data/processed/matched_dataset.csv")
print(f"Loaded {len(matched_df)} matches")
print(matched_df.head())

# Get first fire-scene match
first_match = matched_df.iloc[0]
print(f"\nFirst match:")
print(f"Fire lat: {first_match['fire_lat']}, lon: {first_match['fire_lon']}")
print(f"Scene: {first_match['scene_id']}")
print(f"Scene path: {first_match['scene_path']}")

# Try to load the scene
scene_path = Path(first_match['scene_path'])
band_file = scene_path / "B04.tif"

print(f"\nBand file: {band_file}")
print(f"Exists: {band_file.exists()}")

if band_file.exists():
    with rasterio.open(band_file) as src:
        print(f"\nRaster info:")
        print(f"  CRS: {src.crs}")
        print(f"  Transform: {src.transform}")
        print(f"  Shape: {src.shape}")
        print(f"  Bounds: {src.bounds}")
        
        # Try to convert fire location to pixel coordinates
        lon, lat = first_match['fire_lon'], first_match['fire_lat']
        print(f"\nConverting fire location ({lon}, {lat}) to pixel coords...")
        
        # First reproject to raster CRS
        xs, ys = reproject_coords('EPSG:4326', src.crs, [lon], [lat])
        x, y = xs[0], ys[0]
        print(f"  Reprojected to {src.crs}: ({x:.2f}, {y:.2f})")
        
        row, col = rowcol(src.transform, x, y)
        print(f"  Pixel row: {row}, col: {col}")
        print(f"  Image shape: {src.shape} (height x width)")
        
        # Check if within bounds
        if 0 <= row < src.shape[0] and 0 <= col < src.shape[1]:
            print(f"  ✓ Fire location is within image bounds")
            
            # Try to extract a 256x256 tile
            tile_size = 256
            half_size = tile_size // 2
            row_start = max(0, row - half_size)
            col_start = max(0, col - half_size)
            
            print(f"\nTile extraction:")
            print(f"  Row range: {row_start} to {row_start + tile_size}")
            print(f"  Col range: {col_start} to {col_start + tile_size}")
            
            if row_start + tile_size <= src.shape[0] and col_start + tile_size <= src.shape[1]:
                print(f"  ✓ Tile fits within image")
                
                # Read the tile
                data = src.read(1)
                tile = data[row_start:row_start+tile_size, col_start:col_start+tile_size]
                
                print(f"\nTile stats:")
                print(f"  Shape: {tile.shape}")
                print(f"  Min: {tile.min()}, Max: {tile.max()}, Mean: {tile.mean():.2f}")
                print(f"  Zeros: {np.sum(tile == 0)}/{tile.size} ({np.sum(tile == 0)/tile.size:.1%})")
            else:
                print(f"  ✗ Tile does NOT fit within image")
        else:
            print(f"  ✗ Fire location is OUTSIDE image bounds")
            print(f"     Row {row} not in [0, {src.shape[0]})")
            print(f"     Col {col} not in [0, {src.shape[1]})")
