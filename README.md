---
title: Portugal Fire Detection
emoji: 🔥
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 8000
pinned: false
license: mit
short_description: Multi-source satellite fire detection demo for Portugal
---

# Portugal Fire Detection

A web app that aggregates active-fire and post-fire data over Portugal
from four public satellite sources and runs a small CNN over Sentinel-2
tiles to flag scenes that look like burning vegetation. Built as a
personal project to learn FastAPI, PostGIS, and on-device remote-sensing
inference end to end.

**Live demo:**
[huggingface.co/spaces/ateodoro/portugal-fires](https://huggingface.co/spaces/ateodoro/portugal-fires)
(read-only, served from a frozen August-2025 snapshot).

## What it does

- Pulls near-real-time active-fire detections from
  [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/) — VIIRS and MODIS
  — country-scoped to Portugal (mainland, Madeira, Açores).
- Overlays Meteosat fire-radiative-power tiles from
  [EUMETSAT](https://view.eumetsat.int/) and Sentinel-3 SLSTR FRP
  granules from Copernicus Data Space.
- Searches the [STAC](https://stacspec.org/) catalog for recent
  Sentinel-2 imagery, useful for post-fire burn-scar inspection.
- Runs a 4-channel MobileViT classifier (≈58 MB checkpoint) over
  user-uploaded or stored tiles and returns a fire probability with a
  small ROC/threshold UI.
- Supports drawn areas-of-interest with email notifications when a fire
  is detected within a polygon.
- Logs an F1 score weekly against a held-out evaluation set so model
  drift is visible over time.

## Stack

- Python 3.12, FastAPI, httpx, APScheduler, SlowAPI rate limiting
- PostGIS via SQLAlchemy 2 async + Alembic
- PyTorch + HuggingFace `transformers` (MobileViT)
- Vanilla JS + Leaflet on the frontend (no build step)
- `uv` for dependency management; everything runs through Docker
  Compose

## Quick start

You need Docker, Docker Compose, and a free
[FIRMS API key](https://firms.modaps.eosdis.nasa.gov/api/).

    cp .env.example .env
    # paste your FIRMS_API_KEY into .env
    docker-compose up --build

Open http://localhost:8000.

EUMETSAT and Copernicus credentials are optional — the app falls back
to public WMS layers and the Element 84 STAC mirror if you don't
provide them.

## Demo mode

Set `DEMO_MODE=true` and the app stops calling external services
entirely. Every endpoint reads from `data/snapshot/`, a frozen bundle
written by `ml/scripts/build_snapshot.py`. AOI writes are blocked, the
scheduler doesn't start, and the model load is skipped.

This is what the Hugging Face Space runs. To regenerate the bundle:

    docker-compose exec web uv run \
      python ml/scripts/build_snapshot.py --start 2025-08-01 --end 2025-08-14

## API

Selected endpoints — see `src/api/` for the full surface.

| Method | Path                                | Purpose                          |
|--------|-------------------------------------|----------------------------------|
| GET    | `/api/health`                       | Liveness + `demo` flag           |
| GET    | `/api/firms/viirs?days=N`           | VIIRS active fires (1–365)       |
| GET    | `/api/firms/modis?days=N`           | MODIS active fires (1–365)       |
| GET    | `/api/firms/history`                | Aggregated historical detections |
| GET    | `/api/eumetsat/frp/latest`          | Latest Meteosat FRP overlay      |
| GET    | `/api/sentinel2/search`             | STAC search for S2 scenes        |
| GET    | `/api/sentinel3/frp`                | S3 SLSTR FRP granule extract     |
| GET    | `/api/inference/model-info`         | Model + training metadata        |
| POST   | `/api/inference/predict`            | Run inference on a tile          |
| GET    | `/api/aoi`, `POST /api/aoi`         | List/create AOIs                 |

## Layout

    src/
      main.py            # app factory, lifespan, CORS, rate limiting
      api/               # route handlers (firms, eumetsat, sentinel,
                         #   sentinel3, aoi, inference, dataset, …)
      services/          # alerts, drift detection
      db/                # SQLAlchemy models + async session
      utils/             # config, http client, demo helpers, geo
      static/            # index.html, app.js, styles.css, leaflet
    ml/
      inference/         # FirePredictor (MobileViT + multimodal)
      models/            # 4-channel MobileViT and dual-encoder S2+S3
      scripts/           # snapshot builder, tile renderer
      training/          # training loops + checkpoints
    data/snapshot/       # frozen demo bundle (only on hf-deploy)

## Development

Run tests inside the container — the project assumes Docker for
everything:

    docker-compose exec web uv run pytest

Add a dependency:

    docker-compose exec web uv add <package>

CI builds the Docker image, runs the suite, and fires a startup smoke
test on every PR. The image build is parameterised by two args:

- `INCLUDE_SNAPSHOT` (default `true`) — bundle `data/` and the
  evaluation-metric JSONs. CI sets this to `false` because origin
  doesn't carry the 190 MB demo snapshot.
- `INCLUDE_MODEL` (default `false`) — keep the 58 MB
  `best_model.pth` checkpoint inside the image. The demo doesn't need
  it; live deploys do.

## Deploy

The Hugging Face Space mirrors a separate `hf-deploy` branch that
bundles `data/snapshot/` (auto-LFS'd). To refresh the demo:

    git checkout hf-deploy
    git merge feat/public-demo               # or main
    # regenerate data/snapshot/ if needed
    git add data/snapshot/
    git commit -m "snapshot: refresh"
    git push hf hf-deploy:main

Origin stays lean — `data/` is gitignored on every other branch, so the
snapshot can never sneak into a non-deploy commit.

## Data sources

All data is freely licensed for non-commercial research and reuse:

- [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/) — VIIRS / MODIS
  active fires
- [EUMETSAT](https://view.eumetsat.int/) — Meteosat FRP-PIXEL
- [Copernicus Data Space](https://dataspace.copernicus.eu/) — Sentinel-2
  L2A, Sentinel-3 SLSTR L2 FRP
- Tile labels for training were derived from FIRMS detections aligned
  to Sentinel-2 patches; see `ml/training/`.

## License

MIT — see `LICENSE`. Satellite data retains its original licensing.
