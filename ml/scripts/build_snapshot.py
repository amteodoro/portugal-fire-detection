#!/usr/bin/env python3
"""Build data/snapshot/ from live APIs + local Postgres + tiles.

Usage:
    docker-compose exec web uv run python ml/scripts/build_snapshot.py \
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


FIRMS_AREA_BASE = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"

# Source variants. NRT covers the last ~2 months only. SP (Standard
# Processing) goes back years but lags ~2 months. We pick automatically
# based on how recent the requested window is.
_FIRMS_SOURCES = {
    "VIIRS": {"nrt": "VIIRS_SNPP_NRT", "sp": "VIIRS_SNPP_SP"},
    "MODIS": {"nrt": "MODIS_NRT",      "sp": "MODIS_SP"},
}


def _firms_source(product: str, end: date) -> str:
    """Pick NRT vs SP based on how old the snapshot's end date is."""
    age_days = (date.today() - end).days
    return _FIRMS_SOURCES[product]["nrt" if age_days <= 60 else "sp"]


async def _firms_window(
    client, api_key: str, source: str, bbox: str, window_end: date, days: int
) -> list[dict]:
    """Fetch one ≤10-day FIRMS window and return parsed GeoJSON features."""
    from src.services.firms_service import _parse_csv  # reuse the CSV parser

    # Endpoint: /area/csv/<key>/<source>/<bbox>/<days>/<date>
    # The date is the END of the window (inclusive). days∈[1,10].
    url = f"{FIRMS_AREA_BASE}/{api_key}/{source}/{bbox}/{days}/{window_end.isoformat()}"
    log.info("FIRMS GET %s/%d/%s", source, days, window_end)
    r = await client.get(url, timeout=30.0)
    r.raise_for_status()
    product_key = "VIIRS" if source.startswith("VIIRS") else "MODIS"
    return _parse_csv(product_key, r.content)


async def fetch_firms(out: Path, start: date, end: date) -> dict:
    """Fetch FIRMS data for the snapshot window per (product, region).

    Walks the date range in ≤10-day windows (FIRMS' per-request cap) and
    auto-selects NRT vs SP based on how old the window is. Requires
    FIRMS_API_KEY in the environment."""
    import httpx
    from src.utils.config import settings
    api_key = settings.FIRMS_API_KEY
    if not api_key:
        log.warning("FIRMS_API_KEY not set — writing empty FeatureCollections")

    out.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}

    async with httpx.AsyncClient() as client:
        for product in ("VIIRS", "MODIS"):
            source = _firms_source(product, end)
            for region, bbox in REGIONS.items():
                features: list[dict] = []
                if api_key:
                    cursor = end
                    while cursor >= start:
                        # FIRMS dated archive endpoint caps days at 5
                        # (the FIRMS docs say 10, but that's only the
                        # un-dated NRT endpoint; with an explicit date
                        # the API rejects anything >5).
                        window_days = min(5, (cursor - start).days + 1)
                        try:
                            chunk = await _firms_window(
                                client, api_key, source, bbox, cursor, window_days
                            )
                            features.extend(chunk)
                        except httpx.HTTPStatusError as exc:
                            log.warning(
                                "FIRMS window failed for %s/%s ending %s: %s",
                                source, region, cursor, exc,
                            )
                        cursor = date.fromordinal(cursor.toordinal() - window_days)
                fc = {"type": "FeatureCollection", "features": features}
                (out / f"{product.lower()}_{region}.geojson").write_text(json.dumps(fc))
                counts[f"{product}_{region}"] = len(features)
                log.info("  → %s/%s: %d features", source, region, len(features))
    return counts


async def fetch_sentinel2(out: Path, start: date, end: date) -> dict:
    """Search S2 over the explicit [start, end] window, copy up to 10
    low-cloud thumbnails locally, and write a manifest with rewritten URLs.

    Bypasses src.services.sentinel_service because that helper computes
    "days back from now", which won't return historical scenes."""
    import httpx
    from datetime import datetime, time, timezone
    from src.utils.config import settings
    from src.utils.geo import parse_bbox

    bbox = parse_bbox(settings.PORTUGAL_BBOX)
    start_iso = datetime.combine(start, time.min, tzinfo=timezone.utc).isoformat()
    end_iso   = datetime.combine(end,   time.max, tzinfo=timezone.utc).isoformat()
    body = {
        "collections": ["sentinel-2-l2a"],
        "bbox": list(bbox),
        "datetime": f"{start_iso}/{end_iso}",
        "limit": 10,
        "query": {"eo:cloud_cover": {"lt": 20}},
        "sortby": [{"field": "properties.datetime", "direction": "desc"}],
    }

    out.mkdir(parents=True, exist_ok=True)
    thumb_dir = out / "thumbnails"
    thumb_dir.mkdir(exist_ok=True)
    items: list[dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.post(f"{settings.STAC_CATALOG_URL}/search", json=body)
        r.raise_for_status()
        features = r.json().get("features", [])
        for f in features:
            assets = f.get("assets", {})
            href = (assets.get("thumbnail") or {}).get("href") or (assets.get("preview") or {}).get("href")
            if not href:
                continue
            local = thumb_dir / f"{f['id']}.jpg"
            try:
                rr = await c.get(href)
                rr.raise_for_status()
                local.write_bytes(rr.content)
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
    log.info("Sentinel-2: %d thumbnails over %s..%s", len(items), start, end)
    return {"sentinel2_items": len(items)}


def _s3_path_to_key(s3_path: str) -> str:
    """Strip the leading '/eodata/' prefix from an OData S3Path so it can
    be used as an S3 object-key prefix on the eodata bucket."""
    p = s3_path.lstrip("/")
    return p[len("eodata/"):] if p.startswith("eodata/") else p


def _extract_fire_pixels_sync(
    s3_client, prefix: str, scene_id: str, sat: str, dt: str,
    bbox: tuple[float, float, float, float],
) -> list[dict]:
    """Download FRP_in.nc + FRP_an.nc from a SLSTR scene's S3 prefix and
    return GeoJSON Features for fire pixels inside `bbox`.

    Synchronous (boto3 + netCDF4 are sync); call from `asyncio.to_thread`."""
    import tempfile
    import netCDF4 as nc

    lon_min, lat_min, lon_max, lat_max = bbox
    features: list[dict] = []

    # Inspect both nadir-views — `in` is the main, `an` covers the
    # alternate stripe. SLSTR products historically vary, so try both
    # and silently skip what's missing or empty.
    for nc_name in ("FRP_in.nc", "FRP_an.nc"):
        key = f"{prefix}{nc_name}"
        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tf:
            tmp_path = tf.name
        try:
            try:
                s3_client.download_file("eodata", key, tmp_path)
            except Exception:
                # File doesn't exist in this product — fine
                continue
            # Some products yield 0-byte FRP files; skip those quietly.
            if Path(tmp_path).stat().st_size < 1024:
                continue
            try:
                ds = nc.Dataset(tmp_path, "r")
            except Exception:
                # Malformed NetCDF — skip
                continue
            try:
                if "latitude" not in ds.variables or "longitude" not in ds.variables:
                    ds.close()
                    continue
                lats = ds.variables["latitude"][:]
                lons = ds.variables["longitude"][:]
                if len(lats) == 0:
                    continue
                frps = ds.variables["FRP_MWIR"][:] if "FRP_MWIR" in ds.variables else None
                frp_unc = ds.variables["FRP_uncertainty_MWIR"][:] if "FRP_uncertainty_MWIR" in ds.variables else None
                day_night = ds.variables["Day_night"][:] if "Day_night" in ds.variables else None
                for i in range(len(lats)):
                    lat, lon = float(lats[i]), float(lons[i])
                    if not (lon_min <= lon <= lon_max and lat_min <= lat <= lat_max):
                        continue
                    props = {
                        "scene_id": scene_id,
                        "satellite": sat,
                        "instrument": f"SLSTR_FRP_{nc_name[:-3]}",
                        "date": dt,
                        "tile_center_lat": lat,
                        "tile_center_lon": lon,
                        "label": 1,
                    }
                    if frps is not None:
                        props["frp"] = float(frps[i])
                    if frp_unc is not None:
                        props["frp_uncertainty"] = float(frp_unc[i])
                    if day_night is not None:
                        props["day_night"] = int(day_night[i])
                    features.append({
                        "type": "Feature",
                        "geometry": {"type": "Point", "coordinates": [lon, lat]},
                        "properties": props,
                    })
            finally:
                ds.close()
        finally:
            try:
                Path(tmp_path).unlink()
            except OSError:
                pass

    return features


async def fetch_sentinel3(out: Path, start: date, end: date, max_scenes: int = 50) -> dict:
    """Fetch Sentinel-3 SLSTR L2 FRP scenes + per-pixel fire detections
    for [start, end] over Portugal.

    1. Catalog: anonymous OData query at catalogue.dataspace.copernicus.eu
    2. Download: Copernicus S3 (eodata.dataspace.copernicus.eu) using
       COPERNICUS_S3_ACCESS_KEY / COPERNICUS_S3_SECRET_KEY.
       We download only the FRP_in.nc / FRP_an.nc files (~150–540 KB
       each), not the full ~70 MB .SEN3 archive.
    3. Parse: netCDF4 reads the lat/lon/FRP arrays, we filter to the
       bbox.

    Falls back to scene-centroid proxies for any scene whose S3 fetch
    or netCDF parse fails. If S3 credentials aren't set, every scene
    becomes a centroid (same behaviour as the anonymous-only path)."""
    import httpx
    from src.utils.config import settings
    from src.utils.geo import parse_bbox

    bbox = parse_bbox(settings.PORTUGAL_BBOX)
    lon_min, lat_min, lon_max, lat_max = bbox
    out.mkdir(parents=True, exist_ok=True)

    # ── 1. Catalog query (anonymous) ──────────────────────────────────
    polygon_wkt = (
        f"POLYGON(({lon_min} {lat_min},{lon_max} {lat_min},"
        f"{lon_max} {lat_max},{lon_min} {lat_max},{lon_min} {lat_min}))"
    )
    odata_filter = (
        "Collection/Name eq 'SENTINEL-3' "
        f"and OData.CSC.Intersects(area=geography'SRID=4326;{polygon_wkt}') "
        f"and ContentDate/Start ge {start.isoformat()}T00:00:00Z "
        f"and ContentDate/Start le {end.isoformat()}T23:59:59Z "
        "and contains(Name, 'SL_2_FRP')"
    )
    odata_url = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
    params = {"$filter": odata_filter, "$top": max_scenes, "$orderby": "ContentDate/Start desc"}

    products: list[dict] = []
    try:
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.get(odata_url, params=params)
            r.raise_for_status()
            products = r.json().get("value", [])
    except Exception as exc:
        log.warning("Sentinel-3 OData search failed: %s", exc)

    # ── 2. Optional S3 client for fire-pixel extraction ───────────────
    s3_client = None
    if settings.COPERNICUS_S3_ACCESS_KEY and settings.COPERNICUS_S3_SECRET_KEY:
        import boto3
        s3_client = boto3.client(
            "s3",
            endpoint_url="https://eodata.dataspace.copernicus.eu",
            aws_access_key_id=settings.COPERNICUS_S3_ACCESS_KEY,
            aws_secret_access_key=settings.COPERNICUS_S3_SECRET_KEY,
            region_name="default",
        )
        log.info("Sentinel-3: Copernicus S3 creds present — extracting fire pixels")
    else:
        log.info("Sentinel-3: no Copernicus S3 creds — using centroid proxies only")

    scenes: list[dict[str, Any]] = []
    fire_features: list[dict[str, Any]] = []

    # ── 3. Per-scene processing (bounded concurrency) ─────────────────
    sem = asyncio.Semaphore(4)

    async def process(prod: dict) -> tuple[dict, list[dict]]:
        name = prod.get("Name", "")
        content_date = prod.get("ContentDate", {})
        dt = (content_date.get("Start") or "")[:10]
        sat = name[:3]  # S3A or S3B
        geom = prod.get("GeoFootprint") or {}
        scene_id = name.replace(".SEN3", "")
        scene_meta = {
            "scene_id": scene_id,
            "satellite": sat,
            "instrument": "SLSTR_FRP",
            "date": dt,
            "size_mb": round((prod.get("ContentLength") or 0) / (1024 * 1024), 1),
            "geometry": geom,
        }

        pixels: list[dict] = []
        if s3_client is not None:
            s3_path = prod.get("S3Path", "")
            prefix = _s3_path_to_key(s3_path).rstrip("/") + "/"
            async with sem:
                try:
                    pixels = await asyncio.to_thread(
                        _extract_fire_pixels_sync,
                        s3_client, prefix, scene_id, sat, dt, bbox,
                    )
                except Exception as exc:
                    log.warning("S3 extract failed for %s: %s", scene_id, exc)
                    pixels = []

        # If no pixels (either no creds, or scene had no fires inside bbox),
        # fall back to a single scene-centroid feature when the centroid
        # itself lies inside the bbox.
        if not pixels and geom.get("coordinates"):
            ring = (
                geom["coordinates"][0][0]
                if geom.get("type") == "MultiPolygon"
                else geom["coordinates"][0]
            )
            if ring:
                lons = [pt[0] for pt in ring]
                lats = [pt[1] for pt in ring]
                cx, cy = sum(lons) / len(lons), sum(lats) / len(lats)
                if lon_min <= cx <= lon_max and lat_min <= cy <= lat_max:
                    pixels = [{
                        "type": "Feature",
                        "geometry": {"type": "Point", "coordinates": [cx, cy]},
                        "properties": {
                            "scene_id": scene_id,
                            "satellite": sat,
                            "instrument": "SLSTR_FRP",
                            "date": dt,
                            "tile_center_lat": cy,
                            "tile_center_lon": cx,
                            "label": 1,
                            "note": "scene-centroid proxy (no fire pixels detected in this scene)",
                        },
                    }]

        return scene_meta, pixels

    if products:
        results = await asyncio.gather(*(process(p) for p in products))
        for scene_meta, pixels in results:
            scenes.append(scene_meta)
            fire_features.extend(pixels)

    (out / "scenes.geojson").write_text(json.dumps(scenes))
    (out / "fire_pixels.geojson").write_text(json.dumps({
        "type": "FeatureCollection", "features": fire_features
    }))
    log.info("Sentinel-3: %d scenes, %d fire pixels (%s) over %s..%s",
             len(scenes), len(fire_features),
             "real" if s3_client else "centroid",
             start, end)
    return {
        "sentinel3_scenes": len(scenes),
        "sentinel3_fires": len(fire_features),
        "sentinel3_mode": "s3_fire_pixels" if s3_client else "centroid_only",
    }


async def fetch_eumetsat(out: Path, start: date, end: date) -> dict:
    """Fetch a historical Meteosat FRP PNG via the EUMETSAT view WMS
    using its time= parameter. Picks 12:00 UTC at the midpoint of the
    snapshot window.

    The default layer name (`msg_fes:fire`) is the public Meteosat
    Second Generation Fire Detection product on view.eumetsat.int — no
    token required."""
    import httpx
    from datetime import datetime, time, timezone, timedelta
    from src.utils.config import settings
    from src.utils.geo import parse_bbox

    wms_url   = settings.EUMETSAT_WMS_URL or "https://view.eumetsat.int/geoserver/wms"
    wms_layer = settings.EUMETSAT_FRP_LAYER or "msg_fes:fire"

    midpoint = start + timedelta(days=(end - start).days // 2)
    timestamp = datetime.combine(midpoint, time(12, 0), tzinfo=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%S.000Z"
    )
    lon_min, lat_min, lon_max, lat_max = parse_bbox(settings.PORTUGAL_BBOX)

    params = {
        "service": "WMS", "request": "GetMap", "version": "1.1.1",
        "layers": wms_layer, "styles": "",
        "format": "image/png", "transparent": "true",
        "srs": "EPSG:4326",
        "bbox": f"{lon_min},{lat_min},{lon_max},{lat_max}",
        "width": 1024, "height": 1024,
        "time": timestamp,
    }

    try:
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.get(wms_url, params=params)
            r.raise_for_status()
            content = r.content
        if len(content) < 1000:
            log.warning("EUMETSAT WMS returned a tiny body (%d bytes) — likely an error response", len(content))
            return {"eumetsat_frp": None}
        target = out / "eumetsat_frp_sample.png"
        target.write_bytes(content)
        log.info("EUMETSAT FRP: %d bytes for %s (layer=%s)", len(content), timestamp, wms_layer)
        return {"eumetsat_frp": target.name, "eumetsat_frp_time": timestamp}
    except Exception as exc:
        log.warning("EUMETSAT WMS fetch failed: %s", exc)
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

    import re as _re
    coord_re = _re.compile(r"^.+_(-?\d+\.\d+)_(-?\d+\.\d+)_(?:positive|negative)_\d+$")

    tiles = []
    for npy in npys:
        tile_id = npy.stem
        arr = np.load(npy)
        np.save(out / f"{tile_id}.npy", arr)

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

        result = predictor.predict_from_array(arr)
        m = coord_re.match(tile_id)
        lat = float(m.group(1)) if m else 0.0
        lon = float(m.group(2)) if m else 0.0
        is_pos = "positive" in tile_id
        is_fire = bool(result.is_fire)
        tiles.append({
            "tile_id": tile_id,
            "lat": lat,
            "lon": lon,
            "fire_probability": round(float(result.fire_probability), 4),
            "is_fire": is_fire,
            "y_true": 1 if is_pos else 0,
            "outcome": (
                "tp" if (is_fire and is_pos) else
                "fp" if (is_fire and not is_pos) else
                "fn" if (not is_fire and is_pos) else
                "tn"
            ),
            "scene_date": None,
            "cloud_cover": None,
        })
    (out / "predictions.json").write_text(json.dumps({"tiles": tiles}))
    return {"tiles": len(tiles)}


async def build_history_sqlite(out: Path) -> dict:
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
    except Exception as exc:
        log.warning("history dump from Postgres failed (%s); leaving empty", exc)
    conn.commit()
    conn.close()
    return {"history_rows": inserted}


def write_aois(out: Path) -> dict:
    """Write a single demo AOI (Aveiro region polygon)."""
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

    counts: dict[str, Any] = {}
    counts.update(await fetch_firms(args.out / "firms", args.start, args.end))
    counts.update(await fetch_sentinel2(args.out / "sentinel2", args.start, args.end))
    counts.update(await fetch_sentinel3(args.out / "sentinel3", args.start, args.end))
    counts.update(await fetch_eumetsat(args.out, args.start, args.end))
    counts.update(build_tiles(args.out / "tiles"))
    counts.update(await build_history_sqlite(args.out))
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
