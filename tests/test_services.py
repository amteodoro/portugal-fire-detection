"""
Unit tests for service modules.

Run with: docker-compose exec web uv run pytest tests/test_services.py
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock


class TestFirmsService:
    """Tests for src/services/firms_service.py"""

    def test_unify_confidence_viirs_high(self):
        """Test VIIRS high confidence mapping."""
        from src.services.firms_service import _unify_confidence

        result = _unify_confidence("VIIRS", "high")
        assert result == 90.0

    def test_unify_confidence_viirs_nominal(self):
        """Test VIIRS nominal confidence mapping."""
        from src.services.firms_service import _unify_confidence

        result = _unify_confidence("VIIRS", "nominal")
        assert result == 60.0

    def test_unify_confidence_viirs_low(self):
        """Test VIIRS low confidence mapping."""
        from src.services.firms_service import _unify_confidence

        result = _unify_confidence("VIIRS", "low")
        assert result == 20.0

    def test_unify_confidence_modis_numeric(self):
        """Test MODIS numeric confidence."""
        from src.services.firms_service import _unify_confidence

        result = _unify_confidence("MODIS", "85")
        assert result == 85.0

    def test_unify_confidence_invalid(self):
        """Test invalid confidence returns None."""
        from src.services.firms_service import _unify_confidence

        result = _unify_confidence("VIIRS", "invalid")
        # Should return default value for unknown VIIRS confidence
        assert result == 50.0

    def test_parse_csv_valid_viirs(self, sample_firms_csv):
        """Test parsing valid VIIRS CSV data."""
        from src.services.firms_service import _parse_csv

        features = _parse_csv("VIIRS", sample_firms_csv)

        assert len(features) == 3
        assert features[0]["type"] == "Feature"
        assert features[0]["geometry"]["type"] == "Point"
        assert features[0]["geometry"]["coordinates"] == [-7.5, 39.5]
        assert features[0]["properties"]["source"] == "VIIRS"
        assert features[0]["properties"]["confidence"] == 90.0  # high
        assert features[0]["properties"]["frp"] == 45.2

    def test_parse_csv_valid_modis(self, sample_firms_modis_csv):
        """Test parsing valid MODIS CSV data."""
        from src.services.firms_service import _parse_csv

        features = _parse_csv("MODIS", sample_firms_modis_csv)

        assert len(features) == 2
        assert features[0]["properties"]["source"] == "MODIS"
        assert features[0]["properties"]["confidence"] == 85.0
        assert features[1]["properties"]["confidence"] == 70.0

    def test_parse_csv_empty(self, sample_firms_empty_csv):
        """Test parsing empty CSV returns empty list."""
        from src.services.firms_service import _parse_csv

        features = _parse_csv("VIIRS", sample_firms_empty_csv)
        assert features == []

    def test_parse_csv_error_response(self, sample_firms_error_response):
        """Test parsing error response returns empty list."""
        from src.services.firms_service import _parse_csv

        features = _parse_csv("VIIRS", sample_firms_error_response)
        assert features == []

    def test_parse_csv_malformed_rows(self):
        """Test parsing CSV with malformed rows skips them."""
        from src.services.firms_service import _parse_csv

        csv_data = b"""latitude,longitude,acq_date,acq_time,confidence,frp,track,scan
invalid,data,here,now,bad,0.0,1,2
39.5,-7.5,2024-08-15,1230,high,45.2,123,456"""

        features = _parse_csv("VIIRS", csv_data)
        # Should only have 1 valid feature
        assert len(features) == 1
        assert features[0]["geometry"]["coordinates"] == [-7.5, 39.5]

    @pytest.mark.asyncio
    async def test_fetch_bbox_no_api_key(self):
        """Test fetch_bbox returns empty when no API key."""
        from src.services.firms_service import fetch_bbox

        with patch("src.services.firms_service.settings") as mock_settings:
            mock_settings.FIRMS_API_KEY = None
            mock_settings.PORTUGAL_BBOX = "-9.5,36.8,-6.0,42.3"

            result = await fetch_bbox("VIIRS", days=1)

            assert result == {"type": "FeatureCollection", "features": []}

    @pytest.mark.asyncio
    async def test_fetch_bbox_with_cache(self, sample_geojson_feature_collection):
        """Test fetch_bbox returns cached data."""
        from src.services.firms_service import fetch_bbox

        with patch("src.services.firms_service.settings") as mock_settings, \
             patch("src.services.firms_service.cache_get") as mock_cache_get:
            mock_settings.FIRMS_API_KEY = "test_key"
            mock_settings.PORTUGAL_BBOX = "-9.5,36.8,-6.0,42.3"
            mock_settings.FIRMS_TTL = 900
            mock_cache_get.return_value = sample_geojson_feature_collection

            result = await fetch_bbox("VIIRS", days=1)

            assert result == sample_geojson_feature_collection
            mock_cache_get.assert_called_once()


class TestSentinelService:
    """Tests for src/services/sentinel_service.py"""

    @pytest.mark.asyncio
    async def test_search_items_with_cache(self, sample_stac_search_response):
        """Test search_items returns cached data."""
        from src.services.sentinel_service import search_items

        with patch("src.services.sentinel_service.file_cache_load") as mock_cache:
            mock_cache.return_value = sample_stac_search_response

            result = await search_items(days=7, cloud_pct=20, limit=12)

            assert result == sample_stac_search_response
            mock_cache.assert_called_once()

    @pytest.mark.asyncio
    async def test_fetch_thumbnail_no_assets(self):
        """Test fetch_thumbnail returns empty dict when no assets."""
        from src.services.sentinel_service import fetch_thumbnail

        item = {"id": "test", "assets": {}}
        result = await fetch_thumbnail(item)

        assert result == {}

    @pytest.mark.asyncio
    async def test_fetch_thumbnail_no_thumbnail_asset(self):
        """Test fetch_thumbnail returns empty dict when no thumbnail asset."""
        from src.services.sentinel_service import fetch_thumbnail

        item = {
            "id": "test",
            "assets": {
                "visual": {"href": "https://example.com/visual.tif"}
            }
        }
        result = await fetch_thumbnail(item)

        assert result == {}


class TestFirmsServiceProducts:
    """Tests for FIRMS service product mapping."""

    def test_products_mapping(self):
        """Test PRODUCTS dictionary contains expected values."""
        from src.services.firms_service import PRODUCTS

        assert "VIIRS" in PRODUCTS
        assert "MODIS" in PRODUCTS
        assert PRODUCTS["VIIRS"] == "VIIRS_SNPP_NRT"
        assert PRODUCTS["MODIS"] == "MODIS_NRT"


import json
from pathlib import Path

import pytest


@pytest.fixture
def fake_snapshot(tmp_path, monkeypatch):
    """Create a minimal snapshot directory and point SNAPSHOT_DIR at it."""
    snap = tmp_path / "snapshot"
    (snap / "firms").mkdir(parents=True)
    (snap / "sentinel2/thumbnails").mkdir(parents=True)
    (snap / "sentinel3").mkdir(parents=True)
    (snap / "tiles").mkdir(parents=True)

    (snap / "manifest.json").write_text(json.dumps({
        "frozen_at": "2026-05-06T10:00:00Z",
        "snapshot_range": {"start": "2024-08-12", "end": "2024-08-19"},
        "regions": ["mainland"],
        "fire_count": {"viirs": 2, "modis": 1, "sentinel3": 0},
        "tile_count": 1,
        "model_version": "best_model.pth",
    }))
    (snap / "firms/viirs_mainland.geojson").write_text(json.dumps({
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature",
             "geometry": {"type": "Point", "coordinates": [-7.5, 39.5]},
             "properties": {"source": "VIIRS", "frp": 12.3,
                            "confidence": 90, "acq_time": "2024-08-15T12:30:00Z",
                            "id": "v1"}},
        ],
    }))
    (snap / "tiles/predictions.json").write_text(json.dumps({
        "tiles": [
            {"tile_id": "tile_001", "lat": 39.5, "lon": -7.5,
             "fire_probability": 0.97, "is_fire": True, "y_true": 1,
             "outcome": "tp", "scene_date": "2024-08-15", "cloud_cover": 5.0},
        ],
    }))
    (snap / "aois.json").write_text(json.dumps([
        {"id": "00000000-0000-0000-0000-000000000001",
         "name": "Demo AOI - Aveiro region",
         "email": None,
         "created_at": "2024-08-10T00:00:00Z",
         "active": True,
         "boundary": {"type": "Polygon",
                      "coordinates": [[[-9, 39], [-8, 39], [-8, 40], [-9, 40], [-9, 39]]]}},
    ]))

    monkeypatch.setenv("SNAPSHOT_DIR", str(snap))
    import importlib
    from src.utils import config as cfg
    importlib.reload(cfg)
    from src.services import snapshot as snap_mod
    importlib.reload(snap_mod)
    return snap_mod


def test_read_manifest(fake_snapshot):
    m = fake_snapshot.read_manifest()
    assert m["snapshot_range"]["start"] == "2024-08-12"


def test_read_firms_returns_geojson(fake_snapshot):
    fc = fake_snapshot.read_firms("VIIRS", "mainland")
    assert fc["type"] == "FeatureCollection"
    assert len(fc["features"]) == 1
    assert fc["features"][0]["properties"]["source"] == "VIIRS"


def test_read_firms_missing_file_returns_empty(fake_snapshot):
    fc = fake_snapshot.read_firms("MODIS", "azores")
    assert fc == {"type": "FeatureCollection", "features": []}


def test_read_predictions(fake_snapshot):
    preds = fake_snapshot.read_predictions()
    assert len(preds["tiles"]) == 1
    assert preds["tiles"][0]["tile_id"] == "tile_001"


def test_find_nearest_prediction(fake_snapshot):
    pred = fake_snapshot.find_nearest_prediction(lat=39.51, lon=-7.51)
    assert pred["tile_id"] == "tile_001"


def test_find_nearest_prediction_empty(fake_snapshot, tmp_path):
    # Wipe predictions and re-read
    (tmp_path / "snapshot/tiles/predictions.json").write_text('{"tiles": []}')
    import importlib
    from src.services import snapshot as snap_mod
    importlib.reload(snap_mod)
    assert snap_mod.find_nearest_prediction(lat=0, lon=0) is None


def test_read_aois(fake_snapshot):
    aois = fake_snapshot.read_aois()
    assert len(aois) == 1
    assert aois[0]["name"].startswith("Demo AOI")
