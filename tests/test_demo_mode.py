"""End-to-end checks that DEMO_MODE=true rewires every public surface
to read from data/snapshot/ instead of external APIs."""
import importlib
import json
from pathlib import Path
import sys

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def demo_app(tmp_path, monkeypatch):
    """Boot the app with DEMO_MODE=true and a controlled snapshot dir."""
    snap = tmp_path / "snapshot"
    (snap / "firms").mkdir(parents=True)
    (snap / "sentinel2/thumbnails").mkdir(parents=True)
    (snap / "sentinel3").mkdir(parents=True)
    (snap / "tiles").mkdir(parents=True)

    (snap / "manifest.json").write_text(json.dumps({
        "snapshot_range": {"start": "2024-08-12", "end": "2024-08-19"},
        "tile_count": 1,
    }))
    (snap / "firms/viirs_mainland.geojson").write_text(json.dumps({
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature",
             "geometry": {"type": "Point", "coordinates": [-7.5, 39.5]},
             "properties": {"source": "VIIRS", "frp": 12.3,
                            "confidence": 90,
                            "acq_time": "2024-08-15T12:30:00Z",
                            "id": "v1"}},
        ],
    }))
    (snap / "firms/modis_mainland.geojson").write_text(json.dumps({
        "type": "FeatureCollection", "features": [],
    }))
    (snap / "tiles/predictions.json").write_text(json.dumps({"tiles": []}))
    (snap / "aois.json").write_text("[]")

    monkeypatch.setenv("DEMO_MODE", "true")
    monkeypatch.setenv("SNAPSHOT_DIR", str(snap))
    monkeypatch.setenv("APP_ENV", "development")  # bypass CORS strictness
    monkeypatch.setenv("CORS_ORIGINS", "http://localhost")

    # Clear module cache for config-dependent modules
    for mod in list(sys.modules.keys()):
        if mod.startswith("src."):
            del sys.modules[mod]

    from src.utils import config as cfg
    from src.services import snapshot as snap_mod
    from src import main as main_mod
    return main_mod.app


def test_firms_viirs_returns_snapshot(demo_app):
    with TestClient(demo_app) as c:
        r = c.get("/api/firms/viirs?days=1")
        assert r.status_code == 200
        body = r.json()
        assert body["type"] == "FeatureCollection"
        assert len(body["features"]) == 1
        assert body["features"][0]["properties"]["id"] == "v1"


def test_firms_modis_returns_empty_snapshot(demo_app):
    with TestClient(demo_app) as c:
        r = c.get("/api/firms/modis?days=1")
        assert r.status_code == 200
        assert r.json()["features"] == []
