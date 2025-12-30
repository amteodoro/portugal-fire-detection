"""
Integration tests for API endpoints.

Run with: docker-compose exec web uv run pytest tests/test_api.py
"""

import pytest
from unittest.mock import patch, AsyncMock


class TestHealthEndpoint:
    """Tests for /api/health endpoint."""

    def test_health_returns_ok(self, app_client):
        """Test health endpoint returns OK status."""
        response = app_client.get("/api/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestConfigEndpoint:
    """Tests for /api/config endpoint."""

    def test_config_returns_bbox_and_center(self, app_client):
        """Test config endpoint returns bbox and center coordinates."""
        response = app_client.get("/api/config")

        assert response.status_code == 200
        data = response.json()

        assert "bbox" in data
        assert "center" in data
        assert "app_name" in data
        assert len(data["bbox"]) == 4
        assert len(data["center"]) == 2

    def test_config_center_is_within_bbox(self, app_client):
        """Test that center is within the bounding box."""
        response = app_client.get("/api/config")
        data = response.json()

        lon_min, lat_min, lon_max, lat_max = data["bbox"]
        center_lat, center_lon = data["center"]

        assert lat_min <= center_lat <= lat_max
        assert lon_min <= center_lon <= lon_max


class TestRootEndpoint:
    """Tests for root endpoint redirect."""

    def test_root_redirects_to_static(self, app_client):
        """Test root endpoint redirects to index.html."""
        response = app_client.get("/", follow_redirects=False)

        assert response.status_code == 307
        assert response.headers["location"] == "/static/index.html"


class TestFirmsEndpoints:
    """Tests for FIRMS API endpoints."""

    def test_viirs_endpoint_exists(self, app_client):
        """Test VIIRS endpoint is accessible."""
        with patch("src.api.firms.fetch_bbox") as mock_fetch:
            mock_fetch.return_value = {
                "type": "FeatureCollection",
                "features": []
            }

            response = app_client.get("/api/firms/viirs?days=1")

            # Should return 200 even with empty features
            assert response.status_code == 200
            data = response.json()
            assert data["type"] == "FeatureCollection"

    def test_modis_endpoint_exists(self, app_client):
        """Test MODIS endpoint is accessible."""
        with patch("src.api.firms.fetch_bbox") as mock_fetch:
            mock_fetch.return_value = {
                "type": "FeatureCollection",
                "features": []
            }

            response = app_client.get("/api/firms/modis?days=1")

            assert response.status_code == 200
            data = response.json()
            assert data["type"] == "FeatureCollection"

    def test_viirs_returns_features(self, app_client, sample_geojson_feature_collection):
        """Test VIIRS endpoint returns features when available."""
        with patch("src.api.firms.fetch_bbox") as mock_fetch:
            mock_fetch.return_value = sample_geojson_feature_collection

            response = app_client.get("/api/firms/viirs?days=1")

            assert response.status_code == 200
            data = response.json()
            assert len(data["features"]) == 1
            assert data["features"][0]["properties"]["source"] == "VIIRS"


class TestSentinel2Endpoints:
    """Tests for Sentinel-2 API endpoints."""

    def test_search_endpoint_exists(self, app_client):
        """Test Sentinel-2 search endpoint is accessible."""
        with patch("src.api.sentinel.search_items") as mock_search:
            mock_search.return_value = {
                "type": "FeatureCollection",
                "features": []
            }

            response = app_client.get("/api/sentinel2/search?days=7&cloud_pct=20")

            assert response.status_code == 200

    def test_search_with_custom_params(self, app_client, sample_stac_search_response):
        """Test Sentinel-2 search with custom parameters."""
        with patch("src.api.sentinel.search_items") as mock_search:
            mock_search.return_value = sample_stac_search_response

            response = app_client.get("/api/sentinel2/search?days=14&cloud_pct=50&limit=20")

            assert response.status_code == 200
            mock_search.assert_called_once_with(14, 50, 20)


class TestStaticFiles:
    """Tests for static file serving."""

    def test_static_index_exists(self, app_client):
        """Test static index.html is served."""
        response = app_client.get("/static/index.html")

        # Should return 200 if file exists
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_static_css_exists(self, app_client):
        """Test static CSS file is served."""
        response = app_client.get("/static/css/styles.css")

        assert response.status_code == 200
        assert "text/css" in response.headers["content-type"]


class TestCORSHeaders:
    """Tests for CORS configuration."""

    def test_cors_middleware_configured(self, app_client):
        """Test CORS middleware is configured on the app."""
        from fastapi.middleware.cors import CORSMiddleware

        # Access the app from the client
        app = app_client.app

        # Check that CORSMiddleware is in the middleware stack
        middleware_classes = [type(m.cls).__name__ if hasattr(m, 'cls') else type(m).__name__ 
                              for m in app.user_middleware]
        
        assert any('CORSMiddleware' in str(m) for m in app.user_middleware), \
            "CORSMiddleware should be configured"
