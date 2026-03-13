def parse_bbox(bbox_str: str) -> tuple[float, float, float, float]:
    """Parse 'lon_min,lat_min,lon_max,lat_max'. Raises ValueError on invalid input."""
    try:
        parts = [float(p.strip()) for p in bbox_str.split(",")]
    except ValueError:
        raise ValueError("bbox values must be floats separated by commas")
    if len(parts) != 4:
        raise ValueError(f"bbox must have 4 values, got {len(parts)}")
    lon_min, lat_min, lon_max, lat_max = parts
    if not (-180 <= lon_min <= 180 and -180 <= lon_max <= 180):
        raise ValueError("Longitude must be in [-180, 180]")
    if not (-90 <= lat_min <= 90 and -90 <= lat_max <= 90):
        raise ValueError("Latitude must be in [-90, 90]")
    if lon_min >= lon_max:
        raise ValueError(f"lon_min ({lon_min}) must be < lon_max ({lon_max})")
    if lat_min >= lat_max:
        raise ValueError(f"lat_min ({lat_min}) must be < lat_max ({lat_max})")
    return (lon_min, lat_min, lon_max, lat_max)
