"""
Tests for /api/sentinel3 endpoints.

Run with: docker-compose exec web uv run pytest tests/test_sentinel3.py -v
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from src.utils.limiter import limiter


def _make_client() -> TestClient:
    """Minimal FastAPI app with only the sentinel3 router."""
    from src.api import sentinel3

    app = FastAPI()
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.include_router(sentinel3.router, prefix="/api/sentinel3")
    return TestClient(app)


@pytest.fixture(scope="module")
def client():
    return _make_client()


class TestSentinel3ScenesEndpoint:
    def test_returns_empty_list_when_dir_missing(self, client):
        with patch("src.api.sentinel3.S3_ZIP_DIR", Path("/nonexistent/path")):
            resp = client.get("/api/sentinel3/scenes")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_parses_valid_zip_filename(self, client, tmp_path):
        zip_file = tmp_path / "S3A_SL_2_FRP____20240815T113456_20240815T131556_20240816T212015_6059_115_167_1800_LN2_O_NT_004.zip"
        zip_file.write_bytes(b"x" * 1024 * 1024)  # 1 MB
        with patch("src.api.sentinel3.S3_ZIP_DIR", tmp_path):
            resp = client.get("/api/sentinel3/scenes")
        assert resp.status_code == 200
        scenes = resp.json()
        assert len(scenes) == 1
        s = scenes[0]
        assert s["satellite"] == "S3A"
        assert s["instrument"] == "SL_2_FRP"
        assert s["date"] == "2024-08-15"
        assert s["size_mb"] == 1.0

    def test_ignores_non_zip_files(self, client, tmp_path):
        (tmp_path / "readme.txt").write_text("not a zip")
        (tmp_path / "S3A_SL_2_FRP____20240815T113456_rest.zip").write_bytes(b"x" * 512)
        with patch("src.api.sentinel3.S3_ZIP_DIR", tmp_path):
            resp = client.get("/api/sentinel3/scenes")
        assert resp.status_code == 200
        scenes = resp.json()
        # Only the zip should appear
        assert len(scenes) == 1


class TestSentinel3FirePixelsEndpoint:
    def test_returns_empty_fc_when_dir_missing(self, client):
        with patch("src.api.sentinel3.LABELS_DIR", Path("/nonexistent/path")):
            resp = client.get("/api/sentinel3/fire-pixels")
        assert resp.status_code == 200
        body = resp.json()
        assert body["type"] == "FeatureCollection"
        assert body["features"] == []

    def test_returns_only_sentinel3_fire_labels(self, client, tmp_path):
        # sentinel3 fire — should be included
        (tmp_path / "tile_s3_fire.json").write_text(json.dumps({
            "tile_id": "tile_001",
            "scene_id": "S3A_20240815",
            "satellite": "sentinel3",
            "label": 1,
            "tile_center_lat": 39.5,
            "tile_center_lon": -7.5,
        }))
        # sentinel3 no-fire — excluded
        (tmp_path / "tile_s3_nofire.json").write_text(json.dumps({
            "tile_id": "tile_002",
            "scene_id": "S3A_20240815",
            "satellite": "sentinel3",
            "label": 0,
            "tile_center_lat": 39.6,
            "tile_center_lon": -7.6,
        }))
        # sentinel2 fire — excluded
        (tmp_path / "tile_s2_fire.json").write_text(json.dumps({
            "tile_id": "tile_003",
            "scene_id": "S2A_20240815",
            "satellite": "sentinel2",
            "label": 1,
            "tile_center_lat": 39.7,
            "tile_center_lon": -7.7,
        }))
        with patch("src.api.sentinel3.LABELS_DIR", tmp_path):
            resp = client.get("/api/sentinel3/fire-pixels")
        assert resp.status_code == 200
        body = resp.json()
        assert body["type"] == "FeatureCollection"
        assert len(body["features"]) == 1
        feat = body["features"][0]
        assert feat["properties"]["tile_id"] == "tile_001"
        assert feat["geometry"]["coordinates"] == [-7.5, 39.5]

    def test_skips_malformed_json(self, client, tmp_path):
        (tmp_path / "bad.json").write_text("not valid json {{{")
        (tmp_path / "good.json").write_text(json.dumps({
            "tile_id": "tile_good",
            "scene_id": "S3A_20240815",
            "satellite": "sentinel3",
            "label": 1,
            "tile_center_lat": 40.0,
            "tile_center_lon": -8.0,
        }))
        with patch("src.api.sentinel3.LABELS_DIR", tmp_path):
            resp = client.get("/api/sentinel3/fire-pixels")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["features"]) == 1
        assert body["features"][0]["properties"]["tile_id"] == "tile_good"
