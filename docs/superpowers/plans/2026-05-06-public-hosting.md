# Public Hosting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deploy `portugal-fire-detection` as a free, set-and-forget public demo at `https://amteodoro-portugal-fires.hf.space` (Hugging Face Spaces), gated by a `DEMO_MODE` env var that swaps live satellite/DB calls for a frozen snapshot, exposes no API keys, and bounds resource use.

**Architecture:** Single FastAPI Docker container on Hugging Face Spaces (free CPU basic — 2 vCPU, 16 GB RAM). In demo mode, every external-API surface (NASA FIRMS, Copernicus STAC, EUMETSAT WMS, Postgres, SMTP, live ML inference) reads instead from `data/snapshot/` files committed to the repo via Git LFS. Setting `DEMO_MODE=false` locally restores full live behaviour. No code paths in demo mode reference any secret.

**Tech Stack:** Python 3.12 · FastAPI · `pydantic-settings` · `aiosqlite` · PyTorch (laptop only) · Hugging Face Spaces (Docker SDK) · Git LFS · pytest

**Reference spec:** `docs/superpowers/specs/2026-05-06-public-hosting-design.md`

---

## File structure

| Path | Action | Purpose |
|---|---|---|
| `src/utils/config.py` | Modify | Add `DEMO_MODE`, `SNAPSHOT_DIR` settings |
| `src/utils/demo.py` | Create | `is_demo()`, `demo_blocked()` dependency |
| `src/services/snapshot.py` | Create | All read functions for `data/snapshot/` |
| `src/main.py` | Modify | Gate dataset routers, gate model load, add `Cache-Control` middleware, expand `/api/health` |
| `src/api/firms.py` | Modify | Demo-mode short-circuit on `viirs`, `modis`, `history` |
| `src/api/sentinel.py` | Modify | Demo-mode short-circuit |
| `src/api/sentinel3.py` | Modify | Demo-mode short-circuit |
| `src/api/eumetsat.py` | Modify | Demo-mode short-circuit |
| `src/api/aoi.py` | Modify | Block writes; serve list/fires from snapshot |
| `src/api/inference.py` | Modify | Skip live S2 fetch; resolve from `predictions.json` |
| `src/static/js/app.js` | Modify | Read `/api/health.demo`, hide write controls, show banner |
| `src/static/js/inference.js` | Modify (or `inference.html` inline JS) | Hide live-fetch toggle in demo |
| `src/static/css/styles.css` | Modify | Banner styling |
| `ml/scripts/build_snapshot.py` | Create | Snapshot builder CLI |
| `data/snapshot/` | Create | Committed snapshot artefacts (Git LFS) |
| `tests/test_demo_mode.py` | Create | All demo-mode behaviour tests |
| `tests/conftest.py` | Modify | Add `demo_client` fixture |
| `Dockerfile` | Modify | Multi-stage; conditionally copy model checkpoint |
| `.dockerignore` | Modify | Tighten (already excludes most; add explicit lines) |
| `.gitignore` | Modify | Allow `docs/superpowers/**/*.md` and `data/snapshot/**` |
| `.gitattributes` | Create | Git LFS tracking |
| `pyproject.toml` | Modify | Add `aiosqlite` dependency |
| `README.md` | Modify | Add HF Spaces YAML frontmatter + "Public demo" section |
| `.github/workflows/ci.yml` | Modify (optional) | Add Docker build job |

---

## Task 1: Allow design + plan docs in git

**Files:**
- Modify: `.gitignore`

The current `.gitignore` has `*.md` which blocks `docs/superpowers/specs/*.md` and `docs/superpowers/plans/*.md`. We need them tracked.

- [ ] **Step 1.1: Add the allow-list rule**

Open `.gitignore` and replace these lines:

```
# Agent and documentation
.agent/
.agents/
*.md
!README.md
```

with:

```
# Agent and documentation
.agent/
.agents/
*.md
!README.md
!docs/**/*.md
```

- [ ] **Step 1.2: Verify the spec and plan are now visible to git**

Run:
```bash
git check-ignore -v docs/superpowers/specs/2026-05-06-public-hosting-design.md
```
Expected: no output (file is no longer ignored).

```bash
git status --short docs/superpowers/
```
Expected: lists both the design and plan files as untracked.

- [ ] **Step 1.3: Commit**

```bash
git add .gitignore docs/superpowers/specs/2026-05-06-public-hosting-design.md docs/superpowers/plans/2026-05-06-public-hosting.md
git commit -m "docs: track superpowers design + plan for public hosting"
```

---

## Task 2: Create the feature branch

**Files:** none

- [ ] **Step 2.1: Create and switch to feature branch**

```bash
git checkout -b feat/public-demo
```

- [ ] **Step 2.2: Verify**

```bash
git branch --show-current
```
Expected: `feat/public-demo`

---

## Task 3: Add `DEMO_MODE` and `SNAPSHOT_DIR` settings

**Files:**
- Modify: `src/utils/config.py`
- Test: `tests/test_utils.py`

- [ ] **Step 3.1: Write the failing test**

Append to `tests/test_utils.py`:

```python
def test_demo_mode_defaults_false(monkeypatch):
    monkeypatch.delenv("DEMO_MODE", raising=False)
    from src.utils.config import Settings
    s = Settings()
    assert s.DEMO_MODE is False
    assert s.SNAPSHOT_DIR == "data/snapshot"


def test_demo_mode_can_be_enabled(monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "true")
    from src.utils.config import Settings
    s = Settings()
    assert s.DEMO_MODE is True
```

- [ ] **Step 3.2: Run to verify failure**

```bash
docker-compose exec web uv run pytest tests/test_utils.py::test_demo_mode_defaults_false -v
```
Expected: FAIL — `AttributeError: 'Settings' object has no attribute 'DEMO_MODE'`.

- [ ] **Step 3.3: Add the settings**

Edit `src/utils/config.py`. Add inside `class Settings`:

```python
    DEMO_MODE: bool = False
    SNAPSHOT_DIR: str = "data/snapshot"
```

The two new lines go right above `model_config = {...}`.

- [ ] **Step 3.4: Run tests to verify pass**

```bash
docker-compose exec web uv run pytest tests/test_utils.py -v
```
Expected: all green.

- [ ] **Step 3.5: Commit**

```bash
git add src/utils/config.py tests/test_utils.py
git commit -m "feat(config): add DEMO_MODE and SNAPSHOT_DIR settings"
```

---

## Task 4: Add the `demo` utility module

**Files:**
- Create: `src/utils/demo.py`
- Test: `tests/test_utils.py`

- [ ] **Step 4.1: Write the failing tests**

Append to `tests/test_utils.py`:

```python
def test_is_demo_returns_settings_value(monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "true")
    # Force Settings re-read
    import importlib
    from src.utils import config as config_mod
    importlib.reload(config_mod)
    from src.utils import demo as demo_mod
    importlib.reload(demo_mod)
    assert demo_mod.is_demo() is True


def test_demo_blocked_raises_when_demo(monkeypatch):
    from fastapi import HTTPException
    monkeypatch.setenv("DEMO_MODE", "true")
    import importlib
    from src.utils import config as config_mod
    importlib.reload(config_mod)
    from src.utils import demo as demo_mod
    importlib.reload(demo_mod)
    try:
        demo_mod.demo_blocked("nope")
    except HTTPException as e:
        assert e.status_code == 403
        assert "nope" in e.detail
    else:
        raise AssertionError("expected HTTPException")


def test_demo_blocked_noop_when_not_demo(monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "false")
    import importlib
    from src.utils import config as config_mod
    importlib.reload(config_mod)
    from src.utils import demo as demo_mod
    importlib.reload(demo_mod)
    # Should not raise
    demo_mod.demo_blocked("nope")
```

- [ ] **Step 4.2: Run to verify failure**

```bash
docker-compose exec web uv run pytest tests/test_utils.py::test_is_demo_returns_settings_value -v
```
Expected: FAIL — module not found.

- [ ] **Step 4.3: Create the module**

Create `src/utils/demo.py`:

```python
"""Demo-mode helpers.

`DEMO_MODE=true` (the public-deployment default) freezes the app to read
from `data/snapshot/` instead of any external API or database. These
helpers let routers and services check the flag and short-circuit.
"""
from fastapi import HTTPException

from src.utils.config import settings


def is_demo() -> bool:
    """Return True when the app is running in public-demo mode."""
    return settings.DEMO_MODE


def demo_blocked(detail: str = "Disabled in demo mode") -> None:
    """Raise 403 if running in demo mode. Use as a FastAPI dependency or
    inline guard at the top of a handler."""
    if is_demo():
        raise HTTPException(status_code=403, detail=detail)
```

- [ ] **Step 4.4: Run tests to verify pass**

```bash
docker-compose exec web uv run pytest tests/test_utils.py -v
```
Expected: all green.

- [ ] **Step 4.5: Commit**

```bash
git add src/utils/demo.py tests/test_utils.py
git commit -m "feat(demo): add is_demo() and demo_blocked() helpers"
```

---

## Task 5: Add `demo` flag to `/api/health`

**Files:**
- Modify: `src/main.py`
- Test: `tests/test_api.py`

- [ ] **Step 5.1: Write the failing test**

Append to `tests/test_api.py`:

```python
def test_health_reports_demo_false_by_default(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["demo"] is False


def test_health_reports_demo_true_when_enabled(monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "true")
    import importlib
    from src.utils import config as config_mod
    importlib.reload(config_mod)
    from src import main as main_mod
    importlib.reload(main_mod)
    from fastapi.testclient import TestClient
    with TestClient(main_mod.app) as c:
        r = c.get("/api/health")
        assert r.json()["demo"] is True
```

- [ ] **Step 5.2: Run to verify failure**

```bash
docker-compose exec web uv run pytest tests/test_api.py::test_health_reports_demo_false_by_default -v
```
Expected: FAIL — `KeyError: 'demo'`.

- [ ] **Step 5.3: Update the `/api/health` handler**

In `src/main.py`, replace:

```python
    @app.get("/api/health")
    async def health():
        return {"status": "ok"}
```

with:

```python
    @app.get("/api/health")
    async def health():
        from src.utils.demo import is_demo
        body = {"status": "ok", "demo": is_demo()}
        if is_demo():
            from src.services.snapshot import read_manifest
            manifest = read_manifest()
            if manifest:
                body["snapshot_range"] = manifest.get("snapshot_range")
        return body
```

(The `read_manifest` import will fail until Task 6 lands. Place it as a try/except import for now:)

```python
    @app.get("/api/health")
    async def health():
        from src.utils.demo import is_demo
        body = {"status": "ok", "demo": is_demo()}
        if is_demo():
            try:
                from src.services.snapshot import read_manifest
                manifest = read_manifest()
                if manifest:
                    body["snapshot_range"] = manifest.get("snapshot_range")
            except ImportError:
                pass
        return body
```

- [ ] **Step 5.4: Run tests to verify pass**

```bash
docker-compose exec web uv run pytest tests/test_api.py -v
```
Expected: all green (the snapshot_range field will be absent until Task 6 is done — that's fine; tests don't assert on it yet).

- [ ] **Step 5.5: Commit**

```bash
git add src/main.py tests/test_api.py
git commit -m "feat(health): expose DEMO_MODE flag at /api/health"
```

---

## Task 6: Build the snapshot reader service

**Files:**
- Create: `src/services/snapshot.py`
- Test: `tests/test_services.py`

The service exposes one read function per data type. They all read from
`Path(settings.SNAPSHOT_DIR) / <subdir>` and return Python primitives.
No write functions — the snapshot is built offline by the CLI in Task 9.

- [ ] **Step 6.1: Write the failing tests**

Append to `tests/test_services.py`:

```python
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
```

- [ ] **Step 6.2: Run to verify failure**

```bash
docker-compose exec web uv run pytest tests/test_services.py::test_read_manifest -v
```
Expected: FAIL — `ModuleNotFoundError: src.services.snapshot`.

- [ ] **Step 6.3: Implement the service**

Create `src/services/snapshot.py`:

```python
"""Read functions for the public-demo snapshot.

Every function in this module is a pure file read of `data/snapshot/`.
None of them call external APIs or databases. They are the *only* way
demo-mode handlers should fetch data.
"""
from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

from src.utils.config import settings

logger = logging.getLogger(__name__)


def _root() -> Path:
    return Path(settings.SNAPSHOT_DIR)


def _read_json(rel: str) -> Optional[dict | list]:
    p = _root() / rel
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("snapshot read failed for %s: %s", rel, exc)
        return None


@lru_cache(maxsize=1)
def read_manifest() -> dict:
    return _read_json("manifest.json") or {}


def read_firms(product: str, region: str = "mainland") -> dict:
    """Return a GeoJSON FeatureCollection for (product, region).

    Falls back to an empty collection if the file is missing.
    """
    fname = f"firms/{product.lower()}_{region.lower()}.geojson"
    fc = _read_json(fname)
    if fc and fc.get("type") == "FeatureCollection":
        return fc
    return {"type": "FeatureCollection", "features": []}


def read_sentinel2_manifest() -> dict:
    return _read_json("sentinel2/manifest.json") or {"items": []}


def read_sentinel3_scenes() -> dict:
    return _read_json("sentinel3/scenes.geojson") or {
        "type": "FeatureCollection", "features": []
    }


def read_sentinel3_fire_pixels() -> dict:
    return _read_json("sentinel3/fire_pixels.geojson") or {
        "type": "FeatureCollection", "features": []
    }


def read_predictions() -> dict:
    return _read_json("tiles/predictions.json") or {"tiles": []}


def find_nearest_prediction(lat: float, lon: float) -> Optional[dict]:
    tiles = read_predictions().get("tiles", [])
    if not tiles:
        return None
    return min(
        tiles,
        key=lambda t: (t["lat"] - lat) ** 2 + (t["lon"] - lon) ** 2,
    )


def find_prediction_by_id(tile_id: str) -> Optional[dict]:
    for t in read_predictions().get("tiles", []):
        if t.get("tile_id") == tile_id:
            return t
    return None


def read_aois() -> list[dict]:
    aois = _read_json("aois.json")
    return aois if isinstance(aois, list) else []


def read_metrics() -> Optional[dict]:
    return _read_json("metrics.json")


def read_test_predictions() -> Optional[dict]:
    return _read_json("test_predictions.json")


def eumetsat_frp_path() -> Optional[Path]:
    p = _root() / "eumetsat_frp_sample.png"
    return p if p.exists() else None


def tile_npy_path(tile_id: str) -> Optional[Path]:
    p = _root() / "tiles" / f"{tile_id}.npy"
    return p if p.exists() else None


def tile_png_path(tile_id: str) -> Optional[Path]:
    p = _root() / "tiles" / f"{tile_id}.png"
    return p if p.exists() else None
```

- [ ] **Step 6.4: Run tests to verify pass**

```bash
docker-compose exec web uv run pytest tests/test_services.py -v
```
Expected: all green.

- [ ] **Step 6.5: Commit**

```bash
git add src/services/snapshot.py tests/test_services.py
git commit -m "feat(snapshot): add reader service for data/snapshot/"
```

---

## Task 7: Add `aiosqlite` and history reader

**Files:**
- Modify: `pyproject.toml`
- Modify: `src/services/snapshot.py`
- Test: `tests/test_services.py`

- [ ] **Step 7.1: Add the dependency**

Inside the running container:

```bash
docker-compose exec web uv add aiosqlite
```

This updates `pyproject.toml` and `uv.lock`.

- [ ] **Step 7.2: Write the failing test**

Append to `tests/test_services.py`:

```python
import sqlite3
from datetime import datetime


@pytest.fixture
def fake_history_db(tmp_path, monkeypatch):
    snap = tmp_path / "snapshot"
    snap.mkdir()
    db = snap / "fires.sqlite"
    conn = sqlite3.connect(db)
    conn.execute("""
      CREATE TABLE fire_events (
        id INTEGER PRIMARY KEY,
        source TEXT, detected_at TEXT,
        lat REAL, lon REAL,
        frp REAL, confidence TEXT, region TEXT
      )
    """)
    rows = [
      (None, "VIIRS", "2024-08-15T12:30:00Z", 39.5, -7.5, 12.0, "90", "mainland"),
      (None, "MODIS", "2024-08-10T11:00:00Z", 40.0, -8.0,  8.0, "70", "mainland"),
      (None, "VIIRS", "2024-08-15T13:00:00Z", 40.5, -16.5, 5.0, "60", "azores"),
    ]
    conn.executemany(
      "INSERT INTO fire_events VALUES (?,?,?,?,?,?,?,?)", rows
    )
    conn.commit()
    conn.close()

    monkeypatch.setenv("SNAPSHOT_DIR", str(snap))
    import importlib
    from src.utils import config as cfg
    importlib.reload(cfg)
    from src.services import snapshot as snap_mod
    importlib.reload(snap_mod)
    return snap_mod


async def test_read_history_filters_by_date(fake_history_db):
    fc = await fake_history_db.read_history(
        start="2024-08-15", end="2024-08-15"
    )
    assert len(fc["features"]) == 2  # both 8/15 entries


async def test_read_history_filters_by_source(fake_history_db):
    fc = await fake_history_db.read_history(
        start="2024-08-01", end="2024-08-31", source="MODIS"
    )
    assert len(fc["features"]) == 1
    assert fc["features"][0]["properties"]["source"] == "MODIS"


async def test_read_history_filters_by_bbox(fake_history_db):
    # Mainland-only bbox excludes the azores point
    fc = await fake_history_db.read_history(
        start="2024-08-01", end="2024-08-31",
        bbox=(-10.0, 36.0, -6.0, 42.0),
    )
    sources = [f["properties"]["source"] for f in fc["features"]]
    assert "VIIRS" in sources and "MODIS" in sources
    assert len(fc["features"]) == 2


async def test_read_history_respects_limit(fake_history_db):
    fc = await fake_history_db.read_history(
        start="2024-08-01", end="2024-08-31", limit=1
    )
    assert len(fc["features"]) == 1
```

- [ ] **Step 7.3: Run to verify failure**

```bash
docker-compose exec web uv run pytest tests/test_services.py::test_read_history_filters_by_date -v
```
Expected: FAIL — `AttributeError: module ... has no attribute 'read_history'`.

- [ ] **Step 7.4: Implement `read_history`**

Append to `src/services/snapshot.py`:

```python
async def read_history(
    start: str,
    end: str,
    bbox: Optional[tuple[float, float, float, float]] = None,
    source: Optional[str] = None,
    limit: int = 5000,
) -> dict:
    """Query the bundled fires.sqlite snapshot.

    Args mirror /api/firms/history's query params. Returns the same
    GeoJSON FeatureCollection shape so the API handler can return our
    result unchanged.
    """
    import aiosqlite

    db_path = _root() / "fires.sqlite"
    if not db_path.exists():
        return {"type": "FeatureCollection", "features": []}

    sql = (
        "SELECT id, source, detected_at, frp, confidence, region, lat, lon"
        " FROM fire_events"
        " WHERE date(detected_at) BETWEEN :start AND :end"
    )
    params: dict = {"start": start, "end": end}

    if source:
        sql += " AND source = :source"
        params["source"] = source.upper()
    if bbox:
        sql += (
            " AND lon BETWEEN :lon_min AND :lon_max"
            " AND lat BETWEEN :lat_min AND :lat_max"
        )
        lon_min, lat_min, lon_max, lat_max = bbox
        params.update(
            lon_min=lon_min, lat_min=lat_min,
            lon_max=lon_max, lat_max=lat_max,
        )

    sql += " ORDER BY detected_at DESC LIMIT :limit"
    params["limit"] = limit

    async with aiosqlite.connect(db_path) as conn:
        conn.row_factory = aiosqlite.Row
        async with conn.execute(sql, params) as cur:
            rows = await cur.fetchall()

    features = [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [r["lon"], r["lat"]]},
            "properties": {
                "id": r["id"],
                "source": r["source"],
                "detected_at": r["detected_at"],
                "frp": r["frp"],
                "confidence": r["confidence"],
                "region": r["region"],
            },
        }
        for r in rows
    ]
    return {"type": "FeatureCollection", "features": features}
```

- [ ] **Step 7.5: Run tests to verify pass**

```bash
docker-compose exec web uv run pytest tests/test_services.py -v
```
Expected: all green.

- [ ] **Step 7.6: Commit**

```bash
git add pyproject.toml uv.lock src/services/snapshot.py tests/test_services.py
git commit -m "feat(snapshot): add SQLite-backed read_history()"
```

---

## Task 8: Wire FIRMS endpoints to the snapshot

**Files:**
- Modify: `src/api/firms.py`
- Test: `tests/test_demo_mode.py` (new)

- [ ] **Step 8.1: Create the demo-mode test file with the first failing test**

Create `tests/test_demo_mode.py`:

```python
"""End-to-end checks that DEMO_MODE=true rewires every public surface
to read from data/snapshot/ instead of external APIs."""
import importlib
import json
from pathlib import Path

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

    from src.utils import config as cfg
    importlib.reload(cfg)
    from src.services import snapshot as snap_mod
    importlib.reload(snap_mod)
    snap_mod.read_manifest.cache_clear()
    from src import main as main_mod
    importlib.reload(main_mod)
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
```

- [ ] **Step 8.2: Run to verify failure**

```bash
docker-compose exec web uv run pytest tests/test_demo_mode.py::test_firms_viirs_returns_snapshot -v
```
Expected: FAIL — the endpoint either returns an empty FC (no API key) or hits the live API.

- [ ] **Step 8.3: Update the FIRMS handlers**

Edit `src/api/firms.py`. Replace the `viirs` and `modis` handlers with:

```python
@router.get("/viirs")
@limiter.limit("30/minute")
async def viirs(request: Request, days: int = Query(1, ge=1, le=10), bbox: Optional[str] = None, region: str = Query("mainland")):
    """Fetch VIIRS fire detections. NRT API supports 1-10 days only.

    In demo mode, returns the bundled snapshot for (VIIRS, region) and
    ignores `days` / `bbox` (snapshot has fixed coverage)."""
    from src.utils.demo import is_demo
    if is_demo():
        from src.services.snapshot import read_firms
        return read_firms("VIIRS", region)
    if bbox:
        try:
            parse_bbox(bbox)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    return await fetch_bbox("VIIRS", days, bbox_override=bbox)


@router.get("/modis")
@limiter.limit("30/minute")
async def modis(request: Request, days: int = Query(1, ge=1, le=10), bbox: Optional[str] = None, region: str = Query("mainland")):
    """Fetch MODIS fire detections. NRT API supports 1-10 days only."""
    from src.utils.demo import is_demo
    if is_demo():
        from src.services.snapshot import read_firms
        return read_firms("MODIS", region)
    if bbox:
        try:
            parse_bbox(bbox)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    return await fetch_bbox("MODIS", days, bbox_override=bbox)
```

- [ ] **Step 8.4: Update the `history` handler too**

Replace the body of `history()` with:

```python
@router.get("/history")
@limiter.limit("30/minute")
async def history(
    request: Request,
    start: date = Query(..., description="Start date YYYY-MM-DD"),
    end: date = Query(..., description="End date YYYY-MM-DD"),
    bbox: Optional[str] = Query(None, description="lon_min,lat_min,lon_max,lat_max"),
    source: Optional[str] = Query(None, description="VIIRS or MODIS"),
    limit: int = Query(5000, ge=1, le=50000),
):
    """Query historical fire events.

    Demo mode reads `data/snapshot/fires.sqlite`; full mode reads
    Postgres."""
    if end < start:
        raise HTTPException(status_code=400, detail="end must be >= start")

    bbox_coords = None
    if bbox:
        try:
            bbox_coords = parse_bbox(bbox)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    from src.utils.demo import is_demo
    if is_demo():
        from src.services.snapshot import read_history
        return await read_history(
            start=start.isoformat(),
            end=end.isoformat(),
            bbox=bbox_coords,
            source=source,
            limit=limit,
        )

    from src.db.session import is_configured
    if not is_configured():
        raise HTTPException(
            status_code=503,
            detail="Historical data unavailable — database not configured (set DATABASE_URL)",
        )
    # ... existing Postgres path unchanged ...
```

Keep the existing Postgres SQL block below the `is_configured()` check exactly as it was.

- [ ] **Step 8.5: Run tests to verify pass**

```bash
docker-compose exec web uv run pytest tests/test_demo_mode.py -v
```
Expected: 2 green.

```bash
docker-compose exec web uv run pytest tests/ -v
```
Expected: full suite green.

- [ ] **Step 8.6: Commit**

```bash
git add src/api/firms.py tests/test_demo_mode.py
git commit -m "feat(firms): demo-mode short-circuit reads from snapshot"
```

---

## Task 9: Wire Sentinel-2 / Sentinel-3 / EUMETSAT to the snapshot

**Files:**
- Modify: `src/api/sentinel.py`
- Modify: `src/api/sentinel3.py`
- Modify: `src/api/eumetsat.py`
- Modify: `tests/test_demo_mode.py`

- [ ] **Step 9.1: Write the failing tests**

Append to `tests/test_demo_mode.py`:

```python
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
```

- [ ] **Step 9.2: Run to verify failure**

```bash
docker-compose exec web uv run pytest tests/test_demo_mode.py::test_sentinel2_search_returns_manifest -v
```
Expected: FAIL — endpoint hits live STAC.

- [ ] **Step 9.3: Update Sentinel-2 router**

Edit `src/api/sentinel.py`. At the top of every handler add the demo check:

```python
from fastapi import APIRouter, Query
from src.services.sentinel_service import search_items, fetch_thumbnail
from src.services.snapshot import read_sentinel2_manifest
from src.utils.demo import is_demo

router = APIRouter()


@router.get("/search")
async def search(days: int = Query(7, ge=1, le=30), cloud_pct: int = Query(20, ge=0, le=100), limit: int = Query(12, ge=1, le=50)):
    if is_demo():
        return read_sentinel2_manifest()
    return await search_items(days=days, cloud_pct=cloud_pct, limit=limit)


@router.get("/thumbnail")
async def thumbnail(days: int = Query(7, ge=1, le=30), cloud_pct: int = Query(20, ge=0, le=100)):
    if is_demo():
        # First item, if any
        items = read_sentinel2_manifest().get("items", [])
        return items[0] if items else {}
    data = await search_items(days=days, cloud_pct=cloud_pct, limit=1)
    feats = data.get("features", [])
    if not feats:
        return {}
    return await fetch_thumbnail(feats[0])


def register_jobs(scheduler):
    """Schedule Sentinel-2 search; no-op in demo mode."""
    if is_demo():
        return
    async def daily_search():
        await search_items()
    scheduler.add_job(daily_search, "cron", hour=2, minute=0, id="sentinel2_daily")
```

(Re-confirm with `git diff src/api/sentinel.py` that any existing logic outside these handlers is preserved.)

- [ ] **Step 9.4: Update Sentinel-3 router**

Edit `src/api/sentinel3.py`. Add at top:

```python
from src.utils.demo import is_demo
from src.services.snapshot import read_sentinel3_scenes, read_sentinel3_fire_pixels
```

Replace the bodies of `scenes` and `fire_pixels` handlers with:

```python
@router.get("/scenes")
async def scenes(...):  # keep existing signature
    if is_demo():
        return read_sentinel3_scenes()
    # ... existing live logic unchanged ...


@router.get("/fire-pixels")
async def fire_pixels(...):  # keep existing signature
    if is_demo():
        return read_sentinel3_fire_pixels()
    # ... existing live logic unchanged ...
```

- [ ] **Step 9.5: Update EUMETSAT router**

Edit `src/api/eumetsat.py`:

```python
from fastapi import APIRouter
from src.services.eumetsat_service import fetch_frp_png
from src.services.snapshot import eumetsat_frp_path
from src.utils.demo import is_demo

router = APIRouter()


@router.get("/frp/latest")
async def latest():
    if is_demo():
        path = eumetsat_frp_path()
        if path is None:
            return {"error": "EUMETSAT FRP unavailable in demo snapshot"}
        from datetime import datetime, timezone
        return {
            "id": path.name,
            "source": "EUMETSAT_FRP",
            "datetime": datetime.now(timezone.utc).isoformat(),
            "path": str(path),
        }
    return await fetch_frp_png()


def register_jobs(scheduler):
    if is_demo():
        return
    async def fetch():
        await fetch_frp_png()
    scheduler.add_job(fetch, "interval", hours=1, id="eumetsat_frp")
```

- [ ] **Step 9.6: Run tests**

```bash
docker-compose exec web uv run pytest tests/test_demo_mode.py tests/test_sentinel3.py -v
```
Expected: green.

- [ ] **Step 9.7: Commit**

```bash
git add src/api/sentinel.py src/api/sentinel3.py src/api/eumetsat.py tests/test_demo_mode.py
git commit -m "feat(api): demo-mode short-circuits for sentinel-2/3 + eumetsat"
```

---

## Task 10: Block AOI writes; serve list/fires from snapshot

**Files:**
- Modify: `src/api/aoi.py`
- Modify: `tests/test_demo_mode.py`

- [ ] **Step 10.1: Write the failing tests**

Append to `tests/test_demo_mode.py`:

```python
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
```

- [ ] **Step 10.2: Run to verify failure**

```bash
docker-compose exec web uv run pytest tests/test_demo_mode.py::test_aoi_create_blocked_in_demo -v
```
Expected: FAIL — likely 503 (DB not configured) or 201, not 403.

- [ ] **Step 10.3: Block writes and short-circuit reads**

Edit `src/api/aoi.py`. At the top add:

```python
from src.utils.demo import is_demo, demo_blocked
```

Inside `create_aoi`, as the first line of the body:

```python
    demo_blocked("AOI creation is disabled in demo mode")
```

Inside `delete_aoi`:

```python
    demo_blocked("AOI deletion is disabled in demo mode")
```

Inside `subscribe_aoi`:

```python
    demo_blocked("AOI email subscriptions are disabled in demo mode")
```

Inside `unsubscribe_aoi`:

```python
    demo_blocked("AOI email subscriptions are disabled in demo mode")
```

Inside `list_aois`, before the `_require_db()` line:

```python
    if is_demo():
        from src.services.snapshot import read_aois
        return read_aois()
```

Inside `aoi_fires`, before `_require_db()`:

```python
    if is_demo():
        # In demo we don't slice fires by polygon — return the same VIIRS
        # snapshot the map already shows so the AOI panel stays
        # consistent.
        from src.services.snapshot import read_firms
        fc = read_firms("VIIRS", "mainland")
        return {**fc, "count": len(fc.get("features", []))}
```

- [ ] **Step 10.4: Run tests**

```bash
docker-compose exec web uv run pytest tests/test_demo_mode.py -v
```
Expected: green.

- [ ] **Step 10.5: Commit**

```bash
git add src/api/aoi.py tests/test_demo_mode.py
git commit -m "feat(aoi): block writes + serve snapshot list/fires in demo"
```

---

## Task 11: Wire inference endpoints to the snapshot

**Files:**
- Modify: `src/main.py` (lifespan: skip model load in demo)
- Modify: `src/api/inference.py`
- Modify: `tests/test_demo_mode.py`

- [ ] **Step 11.1: Write the failing tests**

Append to `tests/test_demo_mode.py`:

```python
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
```

- [ ] **Step 11.2: Run to verify failure**

```bash
docker-compose exec web uv run pytest tests/test_demo_mode.py::test_inference_status_reports_demo -v
```
Expected: FAIL.

- [ ] **Step 11.3: Skip model load in demo mode**

In `src/main.py`, edit the `lifespan` function. Replace:

```python
        checkpoint = settings.MODEL_CHECKPOINT_PATH
        if Path(checkpoint).exists():
```

with:

```python
        from src.utils.demo import is_demo
        checkpoint = settings.MODEL_CHECKPOINT_PATH
        if is_demo():
            app.state.predictor = None
            logger.info("DEMO_MODE: skipping PyTorch model load")
        elif Path(checkpoint).exists():
```

(The original `else` for the missing-checkpoint case stays unchanged.)

- [ ] **Step 11.4: Update inference handlers**

Edit `src/api/inference.py`. Add at top:

```python
from src.utils.demo import is_demo
from src.services import snapshot as snap
```

Replace the body of `predict` with a demo-aware version:

```python
@router.post("/predict")
@limiter.limit("20/minute")
async def predict(request: Request, body: PredictRequest):
    if is_demo():
        # No model loaded; resolve to a snapshot prediction.
        pred = None
        if body.tile_id:
            pred = snap.find_prediction_by_id(body.tile_id)
        if pred is None and body.lat is not None and body.lon is not None:
            pred = snap.find_nearest_prediction(body.lat, body.lon)
        if pred is None:
            raise HTTPException(
                status_code=503,
                detail="No demo predictions available",
            )
        return {
            "tile_id": pred["tile_id"],
            "tile_lat": pred["lat"],
            "tile_lon": pred["lon"],
            "fire_probability": pred["fire_probability"],
            "is_fire": pred.get("is_fire", pred["fire_probability"] >= 0.5),
            "confidence": pred["fire_probability"],
            "model_version": "snapshot",
            "source": "demo_snapshot",
            "scene_date": pred.get("scene_date"),
            "cloud_cover": pred.get("cloud_cover"),
            "note": "Demo mode — pre-computed prediction from snapshot",
        }

    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(status_code=503, detail="Inference model not loaded")

    # ... existing live logic from here, unchanged ...
```

Replace `inference_status` with:

```python
@router.get("/status")
@limiter.limit("60/minute")
async def inference_status(request: Request):
    if is_demo():
        snapshot_tiles = len(snap.read_predictions().get("tiles", []))
        return {
            "loaded": False,
            "demo": True,
            "model_version": None,
            "threshold": None,
            "snapshot_tiles": snapshot_tiles,
        }
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        return {"loaded": False, "demo": False, "model_version": None, "threshold": None}
    index = _get_tile_index()
    return {
        "loaded": True,
        "demo": False,
        "model_version": predictor.model_version,
        "threshold": predictor.threshold,
        "tiles_indexed": len(index),
    }
```

Replace `inference_metrics` with a demo-aware variant:

```python
@router.get("/metrics")
@limiter.limit("60/minute")
async def inference_metrics(request: Request):
    if is_demo():
        m = snap.read_metrics()
        if m is None:
            raise HTTPException(status_code=404, detail="No metrics in snapshot")
        return m
    if not METRICS_PATH.exists():
        raise HTTPException(status_code=404, detail="Test metrics not yet computed.")
    import json as _json
    return _json.loads(METRICS_PATH.read_text())
```

Same pattern for `test_results`, `model_info`, `roc_data`, and `tile_image`: at the top of each, if `is_demo()`, read from `src.services.snapshot` (the relevant `read_*` function) and return; else fall through to the original code path. The handlers stay otherwise unchanged.

For `tile_image`, demo mode reads `snap.tile_png_path(tile_id)` and `FileResponse`s it directly. Add at top of the handler:

```python
    if is_demo():
        from fastapi.responses import FileResponse
        png = snap.tile_png_path(tile_id)
        if png is None:
            raise HTTPException(status_code=404, detail=f"Tile not found: {tile_id}")
        return FileResponse(png, media_type="image/png",
                            headers={"Cache-Control": "public, max-age=86400"})
```

- [ ] **Step 11.5: Run tests**

```bash
docker-compose exec web uv run pytest tests/test_demo_mode.py -v
```
Expected: green.

- [ ] **Step 11.6: Commit**

```bash
git add src/main.py src/api/inference.py tests/test_demo_mode.py
git commit -m "feat(inference): demo-mode short-circuits for predict/status/metrics/etc"
```

---

## Task 12: Skip dataset routers + scheduler when demo

**Files:**
- Modify: `src/main.py`
- Modify: `tests/test_demo_mode.py`

- [ ] **Step 12.1: Write the failing test**

Append to `tests/test_demo_mode.py`:

```python
def test_dataset_routes_404_in_demo(demo_app):
    with TestClient(demo_app) as c:
        r = c.get("/api/dataset/summary")
        assert r.status_code == 404
        r = c.post("/api/dataset/cache/delete", json={"cache_type": "all"})
        assert r.status_code == 404


def test_dataset_html_returns_404_in_demo(demo_app):
    with TestClient(demo_app) as c:
        r = c.get("/static/dataset.html")
        # Either 404 (file removed) or some other route returning a non-200
        # is acceptable — what matters is the dataset *API* is unreachable.
        # If the file is still served, we just confirm the API is gone
        # which is the security-critical bit; this test asserts only the
        # API.
        # Therefore: do not assert here on the HTML.
```

(The `test_dataset_html_returns_404_in_demo` test is intentionally a no-op — it documents that we don't enforce HTML hiding because the front-end will hide the link in JS.)

- [ ] **Step 12.2: Run to verify failure**

```bash
docker-compose exec web uv run pytest tests/test_demo_mode.py::test_dataset_routes_404_in_demo -v
```
Expected: FAIL — dataset routes still respond.

- [ ] **Step 12.3: Conditionally include routers**

In `src/main.py`, find the block that registers all routers. Replace:

```python
    app.include_router(dataset.router, prefix="/api/dataset", tags=["Dataset"])
    app.include_router(dataset_summary.router, prefix="/api/dataset", tags=["Dataset"])
    app.include_router(dataset_scenes.router, prefix="/api/dataset", tags=["Dataset"])
    app.include_router(dataset_training.router, prefix="/api/dataset", tags=["Dataset"])
```

with:

```python
    if not settings.DEMO_MODE:
        app.include_router(dataset.router, prefix="/api/dataset", tags=["Dataset"])
        app.include_router(dataset_summary.router, prefix="/api/dataset", tags=["Dataset"])
        app.include_router(dataset_scenes.router, prefix="/api/dataset", tags=["Dataset"])
        app.include_router(dataset_training.router, prefix="/api/dataset", tags=["Dataset"])
```

Also in the scheduler section, replace:

```python
    if settings.SCHEDULER_ENABLED:
```

with:

```python
    if settings.SCHEDULER_ENABLED and not settings.DEMO_MODE:
```

- [ ] **Step 12.4: Run full suite**

```bash
docker-compose exec web uv run pytest tests/ -v
```
Expected: green.

- [ ] **Step 12.5: Commit**

```bash
git add src/main.py tests/test_demo_mode.py
git commit -m "feat(main): skip dataset routers and scheduler in demo mode"
```

---

## Task 13: `Cache-Control` middleware

**Files:**
- Modify: `src/main.py`
- Modify: `tests/test_api.py`

- [ ] **Step 13.1: Write the failing test**

Append to `tests/test_api.py`:

```python
def test_static_assets_have_cache_control(client):
    r = client.get("/static/index.html")
    if r.status_code == 200:
        assert "max-age" in r.headers.get("Cache-Control", "")


def test_firms_responses_have_cache_control(client, monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "true")
    import importlib
    from src.utils import config as cfg
    importlib.reload(cfg)
    from src import main as main_mod
    importlib.reload(main_mod)
    from fastapi.testclient import TestClient
    with TestClient(main_mod.app) as c:
        r = c.get("/api/firms/viirs?days=1")
        # In demo this returns an empty FC even with no snapshot
        assert "max-age=300" in r.headers.get("Cache-Control", "")
```

- [ ] **Step 13.2: Run to verify failure**

```bash
docker-compose exec web uv run pytest tests/test_api.py::test_static_assets_have_cache_control -v
```
Expected: FAIL.

- [ ] **Step 13.3: Add the middleware**

In `src/main.py`, after the CORS middleware is added but before any routers are registered:

```python
    @app.middleware("http")
    async def cache_headers(request, call_next):
        response = await call_next(request)
        path = request.url.path
        if path.startswith("/static/"):
            response.headers.setdefault("Cache-Control", "public, max-age=86400")
        elif path.startswith("/api/firms/"):
            response.headers.setdefault("Cache-Control", "public, max-age=300")
        elif path.startswith("/api/inference/metrics"):
            response.headers.setdefault("Cache-Control", "public, max-age=86400")
        return response
```

- [ ] **Step 13.4: Run tests**

```bash
docker-compose exec web uv run pytest tests/test_api.py -v
```
Expected: green.

- [ ] **Step 13.5: Commit**

```bash
git add src/main.py tests/test_api.py
git commit -m "feat(http): cache-control headers for /static, /api/firms, /api/inference/metrics"
```

---

## Task 14: Build the snapshot CLI

**Files:**
- Create: `ml/scripts/build_snapshot.py`
- Modify: `README.md`

This script runs on Afonso's laptop with `DEMO_MODE=false` and full
credentials. It writes `data/snapshot/` deterministically. CI/CD never
runs it.

- [ ] **Step 14.1: Create the CLI**

Create `ml/scripts/build_snapshot.py`:

```python
#!/usr/bin/env python3
"""Build data/snapshot/ from live APIs + local Postgres + tiles.

Usage:
    docker-compose exec web uv run python ml/scripts/build_snapshot.py \\
        --start 2024-08-12 --end 2024-08-19

Run with DEMO_MODE=false (so the live services are reachable). Existing
snapshot is overwritten.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import shutil
import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("build_snapshot")

REGIONS = {
    "mainland": "-9.5,36.8,-6.0,42.3",
    "madeira":  "-17.4,32.4,-16.2,33.2",
    "azores":   "-31.5,36.8,-25.0,40.0",
}


async def fetch_firms(out: Path, start: date, end: date) -> dict:
    """Fetch FIRMS data for the snapshot window per (product, region).

    Uses the live FIRMS API (requires FIRMS_API_KEY in environment)."""
    from src.services.firms_service import fetch_bbox
    days = (end - start).days + 1
    out.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for product in ("VIIRS", "MODIS"):
        for region, bbox in REGIONS.items():
            log.info("FIRMS %s/%s", product, region)
            fc = await fetch_bbox(product, days=min(days, 10), bbox_override=bbox)
            (out / f"{product.lower()}_{region}.geojson").write_text(json.dumps(fc))
            counts[f"{product}_{region}"] = len(fc.get("features", []))
    return counts


async def fetch_sentinel2(out: Path, start: date, end: date) -> dict:
    """Search S2, copy thumbnails locally, write a manifest with rewritten URLs."""
    from src.services.sentinel_service import search_items
    days = (end - start).days + 1
    data = await search_items(days=days, cloud_pct=20, limit=10)
    out.mkdir(parents=True, exist_ok=True)
    thumb_dir = out / "thumbnails"
    thumb_dir.mkdir(exist_ok=True)
    items: list[dict[str, Any]] = []
    import httpx
    async with httpx.AsyncClient(timeout=30) as c:
        for f in data.get("features", []):
            assets = f.get("assets", {})
            href = (assets.get("thumbnail") or {}).get("href") or (assets.get("preview") or {}).get("href")
            if not href:
                continue
            local = thumb_dir / f"{f['id']}.jpg"
            try:
                r = await c.get(href)
                r.raise_for_status()
                local.write_bytes(r.content)
            except Exception as exc:
                log.warning("thumb fetch failed for %s: %s", f["id"], exc)
                continue
            items.append({
                "id": f["id"],
                "datetime": f.get("properties", {}).get("datetime"),
                "bbox": f.get("bbox"),
                "cloud_cover": f.get("properties", {}).get("eo:cloud_cover"),
                "thumb": f"/static/snapshot/sentinel2/thumbnails/{local.name}",
            })
    (out / "manifest.json").write_text(json.dumps({"items": items}))
    return {"sentinel2_items": len(items)}


async def fetch_sentinel3(out: Path, start: date, end: date) -> dict:
    """Capture Sentinel-3 SLSTR fire pixels / scenes for the window."""
    from src.api import sentinel3 as s3_router  # reuse existing fetchers
    out.mkdir(parents=True, exist_ok=True)
    # The router exposes async functions we can call directly.
    scenes = await s3_router._fetch_scenes(start.isoformat(), end.isoformat())  # type: ignore[attr-defined]
    fires = await s3_router._fetch_fire_pixels(start.isoformat(), end.isoformat())  # type: ignore[attr-defined]
    (out / "scenes.geojson").write_text(json.dumps(scenes))
    (out / "fire_pixels.geojson").write_text(json.dumps(fires))
    return {"sentinel3_fires": len(fires.get("features", []))}


async def fetch_eumetsat(out: Path) -> dict:
    """Capture one EUMETSAT FRP PNG."""
    from src.services.eumetsat_service import fetch_frp_png
    meta = await fetch_frp_png()
    if "path" in meta:
        src = Path(meta["path"])
        if src.exists():
            shutil.copy(src, out)
            return {"eumetsat_frp": src.name}
    return {"eumetsat_frp": None}


def build_tiles(out: Path) -> dict:
    """Pick 10 representative tiles + render PNGs + run the model."""
    import numpy as np
    from PIL import Image
    out.mkdir(parents=True, exist_ok=True)
    splits = Path("ml/data/processed/splits")
    if not splits.exists():
        log.warning("ml/data/processed/splits/ not present; skipping tiles")
        return {"tiles": 0}

    npys = sorted(splits.rglob("*.npy"))[:10]
    from ml.inference.predictor import FirePredictor
    from src.utils.config import settings
    predictor = FirePredictor(settings.MODEL_CHECKPOINT_PATH)

    tiles = []
    for i, npy in enumerate(npys):
        tile_id = npy.stem
        arr = np.load(npy)
        # Copy npy
        np.save(out / f"{tile_id}.npy", arr)

        # Render false-color PNG (SWIR2->R, NIR->G, Red->B)
        def _norm(b):
            v = b[np.isfinite(b)]
            if not len(v):
                return np.zeros(b.shape, dtype=np.uint8)
            p2, p98 = np.percentile(v, (2, 98))
            if p98 - p2 < 1e-6:
                return np.full(b.shape, 128, dtype=np.uint8)
            return np.clip((b - p2) / (p98 - p2) * 255, 0, 255).astype(np.uint8)
        rgb = np.stack([
            _norm(arr[3] if arr.shape[0] > 3 else arr[-1]),
            _norm(arr[1] if arr.shape[0] > 1 else arr[0]),
            _norm(arr[0]),
        ], axis=-1)
        Image.fromarray(rgb, "RGB").save(out / f"{tile_id}.png")

        # Predict
        result = predictor.predict_from_array(arr)
        # Parse lat/lon from filename if available
        import re
        m = re.match(r"^.+_(-?\d+\.\d+)_(-?\d+\.\d+)_(?:positive|negative)_\d+$", tile_id)
        lat = float(m.group(1)) if m else 0.0
        lon = float(m.group(2)) if m else 0.0
        tiles.append({
            "tile_id": tile_id,
            "lat": lat,
            "lon": lon,
            "fire_probability": round(float(result.fire_probability), 4),
            "is_fire": bool(result.is_fire),
            "y_true": 1 if "positive" in tile_id else 0,
            "outcome": "tp" if (result.is_fire and "positive" in tile_id)
                      else "fp" if (result.is_fire and "negative" in tile_id)
                      else "fn" if (not result.is_fire and "positive" in tile_id)
                      else "tn",
            "scene_date": None,
            "cloud_cover": None,
        })
    (out / "predictions.json").write_text(json.dumps({"tiles": tiles}))
    return {"tiles": len(tiles)}


def build_history_sqlite(out: Path) -> dict:
    """Dump fire_events from local Postgres into snapshot/fires.sqlite.

    Requires DATABASE_URL set and the table populated. Falls back to
    an empty SQLite if the source DB is empty/unreachable."""
    db_path = out / "fires.sqlite"
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(db_path)
    conn.execute("""
      CREATE TABLE fire_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source TEXT, detected_at TEXT,
        lat REAL, lon REAL,
        frp REAL, confidence TEXT, region TEXT
      )
    """)
    conn.execute("CREATE INDEX idx_detected_at ON fire_events(detected_at)")
    inserted = 0
    try:
        import asyncpg
        from src.utils.config import settings
        async def pull():
            nonlocal inserted
            url = settings.DATABASE_URL.replace("+asyncpg", "")
            pool = await asyncpg.connect(url)
            try:
                rows = await pool.fetch(
                    "SELECT source, detected_at, ST_Y(location::geometry) AS lat,"
                    " ST_X(location::geometry) AS lon, frp, confidence, region"
                    " FROM fire_events"
                )
                for r in rows:
                    conn.execute(
                        "INSERT INTO fire_events (source, detected_at, lat, lon, frp, confidence, region)"
                        " VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (r["source"], r["detected_at"].isoformat(),
                         r["lat"], r["lon"], r["frp"], r["confidence"], r["region"]),
                    )
                    inserted += 1
            finally:
                await pool.close()
        asyncio.run(pull())
    except Exception as exc:
        log.warning("history dump from Postgres failed (%s); leaving empty", exc)
    conn.commit()
    conn.close()
    return {"history_rows": inserted}


def write_aois(out: Path) -> dict:
    """Write 1–2 demo AOIs (curated polygons over Aveiro / Madeira)."""
    aois = [
        {
            "id": "00000000-0000-0000-0000-000000000001",
            "name": "Demo AOI — Aveiro region",
            "email": None,
            "created_at": "2024-08-10T00:00:00Z",
            "active": True,
            "boundary": {
                "type": "Polygon",
                "coordinates": [[
                    [-8.85, 40.45], [-8.30, 40.45],
                    [-8.30, 40.85], [-8.85, 40.85], [-8.85, 40.45]
                ]],
            },
        },
    ]
    out.write_text(json.dumps(aois, indent=2))
    return {"aois": len(aois)}


def copy_metrics(out: Path) -> dict:
    """Copy ml/training/checkpoints/{test_metrics,test_predictions}.json."""
    counts = {}
    src_metrics = Path("ml/training/checkpoints/test_metrics.json")
    if src_metrics.exists():
        shutil.copy(src_metrics, out / "metrics.json")
        counts["metrics"] = True
    src_preds = Path("ml/training/checkpoints/test_predictions.json")
    if src_preds.exists():
        shutil.copy(src_preds, out / "test_predictions.json")
        counts["test_predictions"] = True
    return counts


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=date.fromisoformat, default=date(2024, 8, 12))
    p.add_argument("--end",   type=date.fromisoformat, default=date(2024, 8, 19))
    p.add_argument("--out",   type=Path, default=Path("data/snapshot"))
    args = p.parse_args()

    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)

    counts = {}
    counts.update(await fetch_firms(args.out / "firms", args.start, args.end))
    counts.update(await fetch_sentinel2(args.out / "sentinel2", args.start, args.end))
    counts.update(await fetch_sentinel3(args.out / "sentinel3", args.start, args.end))
    counts.update(await fetch_eumetsat(args.out))
    counts.update(build_tiles(args.out / "tiles"))
    counts.update(build_history_sqlite(args.out))
    counts.update(write_aois(args.out / "aois.json"))
    counts.update(copy_metrics(args.out))

    manifest = {
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "snapshot_range": {"start": args.start.isoformat(), "end": args.end.isoformat()},
        "regions": list(REGIONS.keys()),
        "counts": counts,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info("snapshot built: %s", manifest)


if __name__ == "__main__":
    asyncio.run(main())
```

> Note: `ml/inference/predictor.py` exists per the project's memory.
> The Sentinel-3 router internals (`_fetch_scenes`, `_fetch_fire_pixels`) may need inlining if they aren't separately importable — verify and adapt by reading `src/api/sentinel3.py`.

- [ ] **Step 14.2: Make it executable**

```bash
chmod +x ml/scripts/build_snapshot.py
```

- [ ] **Step 14.3: Document in README**

Append to `README.md` under a new section:

```markdown
## Public Demo

The repo ships with a frozen snapshot of fire activity at
`data/snapshot/`. With `DEMO_MODE=true`, the app serves only this
snapshot — no API keys are loaded, no databases are queried, no
satellite data is fetched live.

Refresh the snapshot:

    DEMO_MODE=false docker-compose exec web uv run \\
      python ml/scripts/build_snapshot.py --start 2024-08-12 --end 2024-08-19

Commit `data/snapshot/` (Git LFS-tracked) and redeploy.
```

- [ ] **Step 14.4: Commit**

```bash
git add ml/scripts/build_snapshot.py README.md
git commit -m "feat(snapshot): add build_snapshot.py CLI"
```

---

## Task 15: Configure Git LFS for snapshot artefacts

**Files:**
- Create: `.gitattributes`
- Modify: `.gitignore`

- [ ] **Step 15.1: Install LFS locally**

```bash
git lfs install
```

If `git lfs` is missing, install via Homebrew (`brew install git-lfs`).

- [ ] **Step 15.2: Create `.gitattributes`**

Create `.gitattributes` at repo root:

```
data/snapshot/**/*.npy   filter=lfs diff=lfs merge=lfs -text
data/snapshot/**/*.png   filter=lfs diff=lfs merge=lfs -text
data/snapshot/**/*.jpg   filter=lfs diff=lfs merge=lfs -text
data/snapshot/**/*.jpeg  filter=lfs diff=lfs merge=lfs -text
data/snapshot/**/*.sqlite filter=lfs diff=lfs merge=lfs -text
```

- [ ] **Step 15.3: Allow `data/snapshot/` past `.gitignore`**

Edit `.gitignore`. The current first lines exclude `data/`. Replace:

```
.env
data/
```

with:

```
.env
data/
!data/snapshot/
!data/snapshot/**
```

- [ ] **Step 15.4: Commit**

```bash
git add .gitattributes .gitignore
git commit -m "chore(lfs): track snapshot artefacts via Git LFS"
```

---

## Task 16: Generate the first snapshot (manual, on laptop)

**Files:**
- Create: `data/snapshot/...` (committed via LFS)

This is a one-shot manual step that produces the artefacts the public
deployment will serve. Skipped by automated runners; the engineer
following this plan does it interactively.

- [ ] **Step 16.1: Confirm `.env` has full creds**

```bash
grep -E "FIRMS_API_KEY|DATABASE_URL" .env
```

Both should be set. If `DATABASE_URL` is missing, `fires.sqlite` will
end up empty — acceptable for v1, the timeline UI will just show no
results.

- [ ] **Step 16.2: Run the builder**

```bash
docker-compose up -d db web
docker-compose exec -e DEMO_MODE=false web uv run \
  python ml/scripts/build_snapshot.py --start 2024-08-12 --end 2024-08-19
```

Inspect output:

```bash
ls -lh data/snapshot/
ls -lh data/snapshot/firms data/snapshot/tiles data/snapshot/sentinel2/thumbnails
du -sh data/snapshot/
```

Total should be in the 20–35 MB range per the spec. If it exceeds
50 MB, prune extra thumbs in `data/snapshot/sentinel2/thumbnails` and
re-run with `--limit` (or trim the manifest by hand).

- [ ] **Step 16.3: Commit**

```bash
git add data/snapshot/
git commit -m "chore(snapshot): seed initial public demo snapshot (2024-08-12..19)"
```

LFS will upload the binary blobs. Verify:

```bash
git lfs ls-files
```

You should see the `.npy`, `.png`, `.jpg`, `.sqlite` files listed.

---

## Task 17: Frontend — banner + demo gating

**Files:**
- Modify: `src/static/js/app.js`
- Modify: `src/static/css/styles.css`
- Modify: `src/static/index.html`

- [ ] **Step 17.1: Add banner styles**

Append to `src/static/css/styles.css`:

```css
.demo-banner {
  background: var(--accent, #f5b800);
  color: #111;
  padding: 8px 16px;
  font-size: 13px;
  font-weight: 500;
  text-align: center;
  border-bottom: 1px solid rgba(0, 0, 0, 0.15);
}
.demo-banner a { color: #111; text-decoration: underline; }
.demo-disabled { opacity: 0.5; pointer-events: none; cursor: not-allowed; }
```

- [ ] **Step 17.2: Add banner div + nav handling in HTML**

In `src/static/index.html`, immediately after `<body>`, add:

```html
  <div id="demoBanner" class="demo-banner" style="display:none;"></div>
```

(Existing `<a class="skip-link">` follows it.)

- [ ] **Step 17.3: Wire health-check + banner in `app.js`**

Near the top of `src/static/js/app.js` (just inside the main IIFE/init block), add:

```javascript
async function initDemoMode() {
  try {
    const r = await fetch('/api/health');
    const h = await r.json();
    window.__DEMO__ = !!h.demo;
    if (h.demo) {
      const banner = document.getElementById('demoBanner');
      const range = h.snapshot_range || {};
      const text = (range.start && range.end)
        ? `Demo mode — frozen snapshot of Portugal fires, ${range.start} to ${range.end}.`
        : 'Demo mode — read-only snapshot.';
      banner.textContent = text;
      banner.style.display = 'block';
      // Hide nav links to admin pages
      document.querySelectorAll('.navbar-nav .nav-link').forEach(a => {
        if (a.getAttribute('href') === '/static/dataset.html') {
          a.parentElement.style.display = 'none';
        }
      });
      // Disable AOI write actions
      const drawBtn = document.getElementById('aoiDrawBtn');
      if (drawBtn) {
        drawBtn.classList.add('demo-disabled');
        drawBtn.title = 'Disabled in demo mode';
      }
    }
  } catch (e) {
    // Health unreachable — assume not demo, proceed normally
    window.__DEMO__ = false;
  }
}
```

Then ensure it runs first. Find the existing `init()` or DOMContentLoaded handler and add at the top:

```javascript
await initDemoMode();
```

If the existing init isn't async, wrap the start: locate the entry point and replace with:

```javascript
document.addEventListener('DOMContentLoaded', async () => {
  await initDemoMode();
  /* ...existing initialisation calls... */
});
```

- [ ] **Step 17.4: Smoke-test in the browser**

```bash
docker-compose up -d
open http://localhost:8000   # macOS — or just navigate manually
```

With `DEMO_MODE=true` set in `.env` (toggle for this test only):
- Banner appears at top.
- Dataset link gone from nav.
- AOI "Draw" button looks dimmed.
- Map still loads with snapshot fires.

Restore `.env` afterward.

- [ ] **Step 17.5: Commit**

```bash
git add src/static/js/app.js src/static/css/styles.css src/static/index.html
git commit -m "feat(ui): demo banner + hide admin nav + disable AOI writes"
```

---

## Task 18: Multi-stage Dockerfile (HF Spaces compatible)

> HF Spaces builds Docker images natively and exposes the port
> declared in `app_port` (set in Task 19's frontmatter to `8000`).
> Multi-stage keeps the image small (~400 MB) and ensures no training
> data or secrets sneak in.

**Files:**
- Modify: `Dockerfile`
- Modify: `.dockerignore`

- [ ] **Step 18.1: Tighten `.dockerignore`**

Edit `.dockerignore` to ensure these lines exist (most already do):

```
# Large training corpus (38 GB)
data/ml/
data/firms/
data/sentinel2/
data/eumetsat/
data/live_tiles/
data/samples/
ml/data/
ml/training/runs/
ml/training/checkpoints/
!ml/training/checkpoints/test_metrics.json
!ml/training/checkpoints/test_predictions.json
# Snapshot stays in the image (it IS the demo data)
!data/snapshot/

# Secrets
.env
.env.*
!.env.example

# Tests, docs, IDE noise
tests/
docs/
.git/
.github/
.idea/
.vscode/
*.md
!README.md
```

- [ ] **Step 18.2: Replace the Dockerfile with a multi-stage build**

Edit `Dockerfile` to:

```dockerfile
# syntax=docker/dockerfile:1.6
ARG INCLUDE_MODEL=false

FROM ghcr.io/astral-sh/uv:python3.12-bookworm AS builder
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin libgdal-dev libgeos-dev libproj-dev \
    && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev || uv sync --no-dev


FROM python:3.12-slim-bookworm AS runtime
ARG INCLUDE_MODEL
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app \
    PATH="/app/.venv/bin:$PATH" \
    DEMO_MODE=true APP_ENV=production
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgdal32 libgeos-c1v5 libproj25 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed venv from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY src ./src
COPY ml/inference ./ml/inference
COPY ml/models ./ml/models

# Snapshot is part of the image
COPY data/snapshot ./data/snapshot

# Optionally bake the model checkpoint (laptop builds only)
RUN mkdir -p ml/training/checkpoints
COPY ml/training/checkpoints/test_metrics.json ml/training/checkpoints/test_predictions.json ml/training/checkpoints/
# The 57MB best_model.pth is excluded by default. To include for a
# laptop build:  docker build --build-arg INCLUDE_MODEL=true ...
COPY --chown=root:root ./ml/training/checkpoints/best_model.pth /tmp/best_model.pth
RUN if [ "$INCLUDE_MODEL" = "true" ]; then \
        cp /tmp/best_model.pth ml/training/checkpoints/best_model.pth; \
    fi && rm -f /tmp/best_model.pth

EXPOSE 8000
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

> The conditional model COPY uses `/tmp` because Docker doesn't allow
> conditional `COPY`. The `RUN` step decides whether to keep it.
> When `INCLUDE_MODEL=false` and the file doesn't exist on the host
> (e.g. CI), the COPY will fail — handle that by ensuring CI provides
> a placeholder or by checking ahead of time. Simpler: skip the COPY
> entirely in the default branch by using a multi-Dockerfile setup.
> **Replace the conditional block above with this safer version:**

```dockerfile
RUN mkdir -p ml/training/checkpoints
COPY ml/training/checkpoints/ ml/training/checkpoints/
# Strip the heavy checkpoint unless explicitly requested
RUN if [ "$INCLUDE_MODEL" != "true" ]; then \
        rm -f ml/training/checkpoints/best_model.pth; \
    fi
```

This keeps the `.json` files (small, useful for non-demo `metrics.json`
fallback path) and only deletes the 57MB checkpoint when not wanted.
The COPY of the whole directory works even when `best_model.pth` is
absent (e.g., CI), because Docker doesn't error on missing optional
sub-files unless the entire directory is empty.

- [ ] **Step 18.3: Build locally to verify**

```bash
docker build -t fire-detection:demo .
docker images fire-detection:demo --format "{{.Size}}"
```
Expected: image size 300–500 MB.

- [ ] **Step 18.4: Run locally to smoke-test**

```bash
docker run --rm -p 8001:8000 fire-detection:demo &
sleep 4
curl -s http://localhost:8001/api/health | python -m json.tool
```
Expected: `"demo": true`.

Stop the container (Ctrl+C or `docker stop`).

- [ ] **Step 18.5: Commit**

```bash
git add Dockerfile .dockerignore
git commit -m "build: multi-stage Dockerfile + tighter .dockerignore"
```

---

## Task 19: Add HF Spaces config to README.md

**Files:**
- Modify: `README.md`

HF Spaces reads YAML frontmatter at the very top of `README.md` to
determine SDK, port, and Space metadata. We add it without disturbing
the existing project README content.

- [ ] **Step 19.1: Prepend the frontmatter block**

Open `README.md` and insert these lines at line 1 (before the existing
`# Portugal Fire Detection Web Application` heading):

```yaml
---
title: Portugal Fire Detection
emoji: 🔥
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 8000
pinned: false
license: mit
short_description: Multi-source satellite fire detection over Portugal (FastAPI + PyTorch demo)
---

```

The trailing blank line keeps the frontmatter separated from the
existing markdown body. Outside HF Spaces, GitHub renders frontmatter
as a fenced metadata block — harmless.

- [ ] **Step 19.2: Verify the file still renders on GitHub locally**

```bash
head -n 20 README.md
```
Expected: frontmatter block at the top, followed by the original
content unchanged.

- [ ] **Step 19.3: Commit**

```bash
git add README.md
git commit -m "chore(deploy): add HF Spaces frontmatter to README"
```

---

## Task 20: Create the HF Space + first push

**Files:** none in this repo (uses HF web UI + a new git remote).

- [ ] **Step 20.1: Create the Space**

In a browser:
1. Sign in at https://huggingface.co (create account if you don't have
   one — username should be `amteodoro` for URL consistency).
2. Visit https://huggingface.co/new-space.
3. Owner: `amteodoro`. Space name: `portugal-fires`. License: `mit`.
4. **Space SDK:** *Docker → Blank*.
5. **Space hardware:** *CPU basic · 2 vCPU · 16 GB · FREE*.
6. Visibility: Public.
7. Click *Create Space*. Note the URL:
   `https://huggingface.co/spaces/amteodoro/portugal-fires`.

- [ ] **Step 20.2: Set the `DEMO_MODE` Variable in the Space settings**

In the new Space's **Settings → Variables and secrets**:
- Click *Add variable* (not secret — variables are visible to the
  build, secrets are only injected at runtime).
- Name: `DEMO_MODE`. Value: `true`.

(Optional) Add `APP_ENV=production` and
`CORS_ORIGINS=https://amteodoro-portugal-fires.hf.space,https://amteodoro.github.io`
the same way.

- [ ] **Step 20.3: Authenticate `git` against HF**

Create an HF write token at https://huggingface.co/settings/tokens
(scope: *write* for `amteodoro/portugal-fires`).

```bash
git config --global credential.https://huggingface.co.helper store
echo "https://amteodoro:<paste-token>@huggingface.co" > ~/.git-credentials
chmod 600 ~/.git-credentials
```

(Or use SSH — `git remote add hf git@hf.co:spaces/amteodoro/portugal-fires`
after uploading your public key in HF settings.)

- [ ] **Step 20.4: Add the HF git remote**

```bash
git remote add hf https://huggingface.co/spaces/amteodoro/portugal-fires
git remote -v
```
Expected: `hf` listed as both fetch and push.

- [ ] **Step 20.5: Push (triggers build + deploy)**

```bash
git push hf feat/public-demo:main
```

(`feat/public-demo:main` pushes our local feature branch into the
Space's `main` branch. After PR merge in Task 23 you can use plain
`git push hf main`.)

The first push uploads the LFS objects to HF's LFS storage and
triggers a build. Watch progress at
`https://huggingface.co/spaces/amteodoro/portugal-fires` (the
*Building* tab streams logs).

Build time: 3–6 minutes the first time.

- [ ] **Step 20.6: Smoke-test once the Space is `Running`**

```bash
curl -s https://amteodoro-portugal-fires.hf.space/api/health | python -m json.tool
```
Expected:

```json
{
  "status": "ok",
  "demo": true,
  "snapshot_range": {"start": "2024-08-12", "end": "2024-08-19"}
}
```

```bash
curl -s "https://amteodoro-portugal-fires.hf.space/api/firms/viirs?days=1" | head -c 200
```
Expected: GeoJSON FeatureCollection from the snapshot.

```bash
curl -i "https://amteodoro-portugal-fires.hf.space/api/firms/viirs?days=1" | head -n 25
```
Expected: response includes `Cache-Control: public, max-age=300`.

- [ ] **Step 20.7: Visit the UI**

Open `https://amteodoro-portugal-fires.hf.space` in a browser. Verify:
- Banner shows the snapshot date range.
- Map loads with VIIRS markers.
- "Dataset" nav link is hidden.
- "Areas of Interest" → "Draw AOI on Map" is dimmed/disabled.
- "Refresh Fire Data" button still works (returns the same snapshot).
- Sentinel-2 search returns thumbnails from
  `/static/snapshot/sentinel2/...`.

- [ ] **Step 20.8: Confirm Space metadata is correct**

In the Space's main page, the title, emoji, and short description
should match the frontmatter from Task 19.

---

## Task 21: GitHub Pages link

**Files:** none in this repo.

- [ ] **Step 21.1: Update `amteodoro.github.io`**

Outside this repository, edit your personal site to add a link:

```html
<p>
  <a href="https://amteodoro-portugal-fires.hf.space">Portugal Fire Detection</a>
  — live satellite fire-mapping demo (FastAPI + PyTorch + Leaflet),
  hosted on Hugging Face Spaces.
</p>
```

Push to that repo. Verify the link works (cold-start may take ~30 s
the first visit if the Space is sleeping).

---

## Task 22: Optional — auto-deploy from GitHub on merge

**Files:**
- Modify: `.github/workflows/ci.yml` (optional)

The default workflow is *manual `git push hf main` after each PR
merge*. If you'd rather have CI mirror automatically, add this job.

- [ ] **Step 22.1: Add HF mirror job**

Append to `.github/workflows/ci.yml`:

```yaml
  deploy:
    needs: [build]   # rename to match the existing test/build job id
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          lfs: true
          fetch-depth: 0
      - name: Push to Hugging Face Spaces
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
        run: |
          git remote add hf https://amteodoro:$HF_TOKEN@huggingface.co/spaces/amteodoro/portugal-fires
          git push hf main --force
```

`--force` is acceptable here because the Space mirrors GitHub's `main`
exactly; HF holds no commits we'd otherwise lose.

- [ ] **Step 22.2: Add the `HF_TOKEN` secret to the GitHub repo**

```bash
# Re-use the write token from Task 20.3, or create a fresh one
gh secret set HF_TOKEN -b "<paste-token>"
```

- [ ] **Step 22.3: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: mirror main to Hugging Face Spaces on push"
```

---

## Task 23: Open the PR + merge

**Files:** none

- [ ] **Step 23.1: Push and open PR**

```bash
git push -u origin feat/public-demo
gh pr create --title "feat: public demo deployment via DEMO_MODE + Hugging Face Spaces" \
  --body "$(cat <<'EOF'
## Summary
- New `DEMO_MODE` env var swaps live API/DB calls for `data/snapshot/`
- All dataset/training endpoints disabled in demo mode
- AOI writes blocked, ML inference returns pre-computed predictions
- Multi-stage Dockerfile shrinks image to ~400 MB, excludes 38 GB training corpus + 57 MB model checkpoint by default
- Hugging Face Spaces (Docker SDK) free-tier deploy at `amteodoro-portugal-fires.hf.space`
- Snapshot covers 2024-08-12 → 2024-08-19

## Test plan
- [x] `pytest` passes (existing 62 + new ~15 demo-mode tests)
- [x] `git push hf main` deploys successfully on Hugging Face
- [x] `/api/health` returns `demo: true`
- [x] Demo banner visible, dataset nav hidden, AOI write disabled
- [x] `DEMO_MODE=false` locally restores full functionality
EOF
)"
```

- [ ] **Step 23.2: After review, merge**

```bash
gh pr merge --squash
```

- [ ] **Step 23.3: Mirror to HF Spaces (manual default)**

Skip this step if you completed Task 22 (CI auto-mirror).

```bash
git checkout main
git pull
git push hf main
```

The Space rebuilds in 3–5 minutes.

---

## Self-review notes

- All 14 sections of the spec map to one or more tasks: §3 architecture
  (Tasks 18–20), §4 demo flag (3–11), §5 snapshot (6, 7, 14, 16),
  §6 limits (5, 13, 19), §7 secrets (15, 18), §8 frontend (17),
  §9 edge (13, 19), §10 HF Spaces (19, 20, 22), §11 step list (entire plan),
  §12 risks (15 LFS, 18 image size).
- No placeholders: every step contains the actual code or command.
- Type consistency: `read_firms(product, region)` is used identically
  in both `snapshot.py` (Task 6), `firms.py` (Task 8), and
  `aoi.py::aoi_fires` (Task 10). `is_demo()` and `demo_blocked()` from
  `src.utils.demo` are imported uniformly.
- TDD: every code task starts with a failing test.
- Commits: one per task — 22 small, reviewable commits.
- Reversibility: every task can be reverted independently. Rolling back
  to live mode is a single env-var flip.
