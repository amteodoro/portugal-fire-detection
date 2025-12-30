"""
Pytest configuration and shared fixtures.

All tests run inside the Docker container via:
    docker-compose exec web uv run pytest
"""

import os
import sys
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from typing import Generator

import pytest
from fastapi.testclient import TestClient

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory) -> Path:
    """Create a temporary directory for test data."""
    return tmp_path_factory.mktemp("test_data")


@pytest.fixture
def sample_firms_csv() -> bytes:
    """Sample FIRMS CSV response data."""
    return b"""latitude,longitude,acq_date,acq_time,confidence,frp,track,scan
39.5,-7.5,2024-08-15,1230,high,45.2,123,456
39.6,-7.6,2024-08-15,1235,nominal,32.1,124,457
39.7,-7.7,2024-08-15,1240,low,12.5,125,458"""


@pytest.fixture
def sample_firms_modis_csv() -> bytes:
    """Sample MODIS FIRMS CSV response data."""
    return b"""latitude,longitude,acq_date,acq_time,confidence,frp,track,scan
40.1,-8.1,2024-08-15,1300,85,55.0,200,300
40.2,-8.2,2024-08-15,1305,70,40.0,201,301"""


@pytest.fixture
def sample_firms_empty_csv() -> bytes:
    """Empty FIRMS CSV response (headers only)."""
    return b"latitude,longitude,acq_date,acq_time,confidence,frp,track,scan\n"


@pytest.fixture
def sample_firms_error_response() -> bytes:
    """FIRMS API error response."""
    return b"error: Invalid API key or request parameters"


@pytest.fixture
def sample_geojson_feature_collection() -> dict:
    """Sample GeoJSON FeatureCollection."""
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [-7.5, 39.5]},
                "properties": {
                    "id": "test_1",
                    "source": "VIIRS",
                    "acq_time": "2024-08-15T12:30:00Z",
                    "confidence": 90.0,
                    "frp": 45.2,
                },
            }
        ],
    }


@pytest.fixture
def sample_stac_search_response() -> dict:
    """Sample STAC search response."""
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "id": "S2A_MSIL2A_20240815T112121_N0510_R037_T29TNE_20240815T143000",
                "bbox": [-9.5, 39.0, -8.5, 40.0],
                "properties": {
                    "datetime": "2024-08-15T11:21:21Z",
                    "eo:cloud_cover": 5.2,
                },
                "assets": {
                    "thumbnail": {
                        "href": "https://example.com/thumbnail.jpg"
                    }
                },
            }
        ],
    }


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    settings = MagicMock()
    settings.FIRMS_API_KEY = "test_api_key"
    settings.FIRMS_TTL = 900
    settings.S2_TTL = 86400
    settings.PORTUGAL_BBOX = "-9.5,36.8,-6.0,42.3"
    settings.STAC_CATALOG_URL = "https://earth-search.aws.element84.com/v1"
    settings.APP_NAME = "Portugal Fire Detection Test"
    settings.CORS_ORIGINS = "*"
    settings.SCHEDULER_ENABLED = False
    settings.TIMEZONE = "UTC"
    return settings


@pytest.fixture
def temp_cache_dir(tmp_path) -> Path:
    """Create temporary cache directory."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return cache_dir


@pytest.fixture
def sample_tile_data() -> dict:
    """Sample tile data for ML dataset tests."""
    import numpy as np
    return {
        "tile": np.random.rand(4, 128, 128).astype(np.float32),
        "label": {
            "tile_id": "test_tile_001",
            "scene_id": "S2A_20240815",
            "label": 1,
            "fire_pixel_row": 64,
            "fire_pixel_col": 64,
        }
    }


@pytest.fixture
def app_client():
    """Create test client for FastAPI app."""
    # Import here to avoid circular imports
    from src.main import create_app
    
    app = create_app()
    with TestClient(app) as client:
        yield client


@pytest.fixture
def mock_http_client():
    """Mock async HTTP client."""
    client = AsyncMock()
    return client
