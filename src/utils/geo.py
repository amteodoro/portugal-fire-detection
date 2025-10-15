def parse_bbox(bbox_str: str) -> tuple[float, float, float, float]:
    parts = [float(p.strip()) for p in bbox_str.split(",")]
    assert len(parts) == 4
    return tuple(parts)  # lon_min, lat_min, lon_max, lat_max
