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

    _reload_app_modules()
    from src import main as main_mod
    yield main_mod.app

    # Teardown: revert env vars, reload modules, and reset the
    # in-memory rate limiter so subsequent tests in the same session
    # start from a clean baseline. monkeypatch teardown runs after
    # this, so we pop env vars explicitly here.
    import os
    for k in ("DEMO_MODE", "SNAPSHOT_DIR", "APP_ENV", "CORS_ORIGINS"):
        os.environ.pop(k, None)
    _reload_app_modules()
    from src.utils.limiter import limiter
    limiter._storage.reset()


# Modules whose `from x import y` bindings depend on config / snapshot
# module identity. Reloaded in dependency order so each one rebinds to
# the freshly-loaded version of its dependencies.
_RELOAD_ORDER = (
    "src.utils.config",
    "src.utils.demo",
    "src.services.snapshot",
    "src.api.firms",
    "src.api.sentinel",
    "src.api.sentinel3",
    "src.api.eumetsat",
    "src.api.aoi",
    "src.api.inference",
    "src.main",
)


def _reload_app_modules() -> None:
    for name in _RELOAD_ORDER:
        if name in sys.modules:
            importlib.reload(sys.modules[name])


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


def test_sentinel2_search_returns_manifest(demo_app, tmp_path):
    snap = Path(tmp_path / "snapshot")
    (snap / "sentinel2/manifest.json").write_text(json.dumps({
        "items": [{"id": "S2_demo_1", "datetime": "2024-08-15", "thumb": "/static/snapshot/sentinel2/thumbnails/s2_demo_1.jpg"}]
    }))
    with TestClient(demo_app) as c:
        r = c.get("/api/sentinel2/search?days=7&cloud_pct=20")
        assert r.status_code == 200
        body = r.json()
        assert body["items"][0]["id"] == "S2_demo_1"


def test_sentinel3_fire_pixels_returns_geojson(demo_app, tmp_path):
    (Path(tmp_path) / "snapshot/sentinel3/fire_pixels.geojson").write_text(json.dumps({
        "type": "FeatureCollection", "features": []
    }))
    with TestClient(demo_app) as c:
        r = c.get("/api/sentinel3/fire-pixels")
        assert r.status_code == 200
        assert r.json()["type"] == "FeatureCollection"


def test_eumetsat_frp_returns_sample_when_present(demo_app, tmp_path):
    sample = Path(tmp_path) / "snapshot/eumetsat_frp_sample.png"
    sample.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\0" * 100)  # minimal PNG bytes
    with TestClient(demo_app) as c:
        r = c.get("/api/eumetsat/frp/latest")
        assert r.status_code == 200
        body = r.json()
        assert body["source"] == "EUMETSAT_FRP"
        assert "path" in body


def test_aoi_create_blocked_in_demo(demo_app):
    with TestClient(demo_app) as c:
        r = c.post("/api/aoi", json={
            "name": "x", "boundary": {
                "type": "Polygon",
                "coordinates": [[[-9, 39], [-8, 39], [-8, 40], [-9, 40], [-9, 39]]],
            }
        })
        assert r.status_code == 403
        assert "demo" in r.json()["detail"].lower()


def test_aoi_subscribe_blocked_in_demo(demo_app):
    with TestClient(demo_app) as c:
        r = c.post("/api/aoi/00000000-0000-0000-0000-000000000001/subscribe",
                   json={"email": "x@y.com"})
        assert r.status_code == 403


def test_aoi_list_returns_snapshot_aois(demo_app, tmp_path):
    aois_path = Path(tmp_path) / "snapshot/aois.json"
    aois_path.write_text(json.dumps([
        {"id": "00000000-0000-0000-0000-000000000001",
         "name": "Demo AOI", "email": None,
         "created_at": "2024-08-10T00:00:00Z", "active": True,
         "boundary": {"type": "Polygon",
                      "coordinates": [[[-9, 39], [-8, 39], [-8, 40], [-9, 40], [-9, 39]]]}}
    ]))
    with TestClient(demo_app) as c:
        r = c.get("/api/aoi")
        assert r.status_code == 200
        body = r.json()
        assert len(body) == 1
        assert body[0]["name"] == "Demo AOI"


def test_inference_status_reports_demo(demo_app):
    with TestClient(demo_app) as c:
        r = c.get("/api/inference/status")
        body = r.json()
        assert body["loaded"] is False
        assert body["demo"] is True


def test_inference_predict_uses_snapshot(demo_app, tmp_path):
    preds = {"tiles": [
        {"tile_id": "tile_001", "lat": 39.5, "lon": -7.5,
         "fire_probability": 0.97, "is_fire": True, "y_true": 1,
         "outcome": "tp", "scene_date": "2024-08-15", "cloud_cover": 5.0},
    ]}
    (Path(tmp_path) / "snapshot/tiles/predictions.json").write_text(json.dumps(preds))
    with TestClient(demo_app) as c:
        r = c.post("/api/inference/predict", json={
            "lat": 39.5, "lon": -7.5, "live": True,
        })
        assert r.status_code == 200
        body = r.json()
        assert body["tile_id"] == "tile_001"
        assert body["fire_probability"] == 0.97
        assert body["source"] == "demo_snapshot"


def test_inference_metrics_reads_snapshot(demo_app, tmp_path):
    (Path(tmp_path) / "snapshot/metrics.json").write_text(json.dumps({
        "f1": 0.985, "auc": 0.996,
    }))
    with TestClient(demo_app) as c:
        r = c.get("/api/inference/metrics")
        assert r.status_code == 200
        assert r.json()["f1"] == 0.985


def test_dataset_routes_404_in_demo(demo_app):
    with TestClient(demo_app) as c:
        r = c.get("/api/dataset/summary")
        assert r.status_code == 404
        r = c.post("/api/dataset/cache/delete", json={"cache_type": "all"})
        assert r.status_code == 404
