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
