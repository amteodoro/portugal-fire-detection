# Public Hosting Design — Portugal Fire Detection

**Status:** Draft, awaiting review
**Author:** Claude (with Afonso)
**Date:** 2026-05-06
**Branch:** `feat/public-demo` (to be created)

## 1. Goal

Make the Portugal Fire Detection web app reachable from a link on
`amteodoro.github.io` as a **set-and-forget portfolio piece** that:

1. Costs €0/month.
2. Reveals **no** API keys, SMTP credentials, database URLs, or
   private model artefacts.
3. Bounds satellite-data egress so the app cannot rack up bills or
   exhaust NASA/Copernicus/EUMETSAT quotas, no matter how many
   visitors arrive.
4. Lets a visitor explore every public-facing feature against a
   curated *snapshot* of fire activity — without making the public
   server perform any expensive live work.
5. Is **fully reversible** locally: the same codebase, run with
   `DEMO_MODE=false` on Afonso's machine, behaves exactly like today.

## 2. Non-goals

- No public access to the dataset/training dashboard or any endpoint
  that spawns subprocesses.
- No real-time AOI email subscriptions for visitors (abuse vector).
- No live ML inference for visitors in v1 (S3 egress per click).
- No multi-user auth, accounts, or persistence of visitor input.
- No CDN in front of the app in v1 (no registered domain to put on
  Cloudflare's free tier).

## 3. Architecture

```
amteodoro.github.io  ──── link ────►   https://amteodoro-portugal-fires.hf.space
                                                  │
                                                  │  HTTPS (HF-issued cert)
                                                  ▼
                                  ┌──────────────────────────────────┐
                                  │  Hugging Face Spaces (Docker SDK)│
                                  │  free hardware: 2 vCPU · 16 GB   │
                                  │  sleeps after ~48 h idle         │
                                  │                                  │
                                  │  Docker (FastAPI)                │
                                  │  - serves /static                │
                                  │  - serves /api/*                 │
                                  │  - reads data/snapshot           │
                                  │  - DEMO_MODE=true                │
                                  └──────────────────────────────────┘
                                                  │
                                                  ▼
                                       ┌───────────────────────┐
                                       │  data/snapshot/       │  (HF Git LFS)
                                       │  ├── manifest.json    │
                                       │  ├── firms/*.geojson  │
                                       │  ├── sentinel2/...    │
                                       │  ├── sentinel3/...    │
                                       │  ├── tiles/*.npy+png  │
                                       │  ├── predictions.json │
                                       │  └── fires.sqlite     │
                                       └───────────────────────┘
```

The Space is connected to the GitHub repo via an HF git remote. Pushing
to `main` triggers an HF rebuild (~3–5 min). HF Spaces handles HTTPS,
caching, and DDoS protection at the platform level; defence-in-depth is
the demo snapshot itself (no live external calls happen on the public
host) plus app-level `slowapi` rate limits.

**No external database, no scheduler, no SMTP, no Copernicus/EUMETSAT
credentials in production.**

### 3.1 Why Hugging Face Spaces?

- **Genuinely free** in 2026 — no credit card, no $5/month minimum
  (Fly.io and Railway both ended their free tiers).
- Generous resources: 2 vCPU + 16 GB RAM means you can flip
  `DEMO_MODE=false` on the hosted instance later (e.g. demo days) and
  load the 57 MB PyTorch checkpoint without trouble.
- Built-in Git LFS with a generous quota — perfect for the snapshot
  artefacts (`.npy`, `.sqlite`, `.png`).
- Free `*.hf.space` subdomain with valid HTTPS, no domain to buy.
- Long sleep window (~48 h idle) so visitors rarely hit a cold start;
  warm-up takes ~30 s when it does happen.
- Auto-deploy from a `git push` to the HF remote — no GitHub Actions
  needed for the demo to redeploy.
- Spaces are public, indexed, and discoverable — extra SEO for a
  portfolio piece.

### 3.2 Why no managed Postgres?

The historical timeline only needs to query an immutable, pre-frozen
snapshot of fire events. SQLite (with the
[`SpatiaLite`](https://www.gaia-gis.it/fossil/libspatialite/index)
extension or plain lat/lon columns + a bbox `WHERE` clause) handles
this in <50 ms for hundreds of thousands of points and ships inside
the image — no external service to provision, no connection string
to leak.

## 4. The `DEMO_MODE` flag

A single environment variable controls behaviour everywhere.

```
# .env (laptop, gitignored)
DEMO_MODE=false                # full live app
FIRMS_API_KEY=…                # real key
DATABASE_URL=postgresql+asyncpg://…
SMTP_USER=…
SMTP_PASS=…

# HF Spaces "Variables" UI / Dockerfile default
DEMO_MODE=true                 # only one var needed
# (no other secrets; demo-mode code paths never call them)
```

### 4.1 Behaviour matrix

| Surface | `DEMO_MODE=true` | `DEMO_MODE=false` |
|---|---|---|
| `/static/dataset.html`, `/api/dataset/*` | Routers not registered (404) | Registered |
| `/static/inference.html` | Served (read-only) | Served |
| `/api/inference/predict` | `live` field forced `false`; **PyTorch model not loaded** in this mode; if `tile_id` matches a snapshot tile or `lat/lon` is near one, return that tile's pre-computed entry from `predictions.json`; otherwise return nearest snapshot tile with a `note: "demo — nearest sample tile"` field | Live S2 COG fetch + CPU inference |
| `/api/inference/status` | `{ loaded: false, demo: true, snapshot_tiles: 10 }` | reports model state |
| `/api/inference/metrics`, `/test-results`, `/roc-data`, `/model-info`, `/tile-image/*` | Served from snapshot files (no model needed) | Live (model loaded) |
| `/api/firms/viirs|modis` | Reads `data/snapshot/firms/{product}_{region}.geojson`; ignores `days` and `bbox` (returns full snapshot, optionally filtered in-process by bbox) | Calls NASA FIRMS |
| `/api/firms/history` | Queries `data/snapshot/fires.sqlite` (bundled, read-only) | Queries Postgres |
| `/api/sentinel2/search`, `/thumbnail` | Returns curated `data/snapshot/sentinel2/manifest.json` | Live STAC |
| `/api/sentinel3/scenes`, `/fire-pixels` | Returns `data/snapshot/sentinel3/*.geojson` | Live |
| `/api/eumetsat/frp/latest` | Returns `data/snapshot/eumetsat_frp_sample.png` | Live WMS |
| `/api/aoi/*` | Read-only list works against snapshot (1–2 demo AOIs); `POST`/`DELETE`/`subscribe` → 403 with friendly JSON | Full CRUD + email |
| `/api/images/{folder}/{filename}` | Path-traversal hardened (resolved path must be inside `/app/data/`); only `samples/` and `snapshot/` exposed | Same hardening + full data dir |
| Scheduler (`SCHEDULER_ENABLED`) | Forced `false` regardless of env | Honoured |
| `/api/health` | `{ "status": "ok", "demo": true, "snapshot_range": "2024-08-12/2024-08-19" }` | `{ "status": "ok", "demo": false }` |

### 4.2 Implementation sketch

```python
# src/utils/demo.py (new)
from fastapi import HTTPException, status
from src.utils.config import settings


def demo_blocked(detail: str = "Disabled in demo mode"):
    """Use as a FastAPI dependency on routes you want disabled."""
    if settings.DEMO_MODE:
        raise HTTPException(status_code=403, detail=detail)


def is_demo() -> bool:
    return settings.DEMO_MODE
```

```python
# src/main.py (excerpt)
if not settings.DEMO_MODE:
    app.include_router(dataset.router, prefix="/api/dataset", tags=["Dataset"])
    app.include_router(dataset_summary.router, prefix="/api/dataset", tags=["Dataset"])
    app.include_router(dataset_scenes.router, prefix="/api/dataset", tags=["Dataset"])
    app.include_router(dataset_training.router, prefix="/api/dataset", tags=["Dataset"])
```

Inside individual handlers (e.g. `aoi.create_aoi`,
`aoi.subscribe_aoi`, `inference.predict` when `body.live=True`), add
the dependency or an early-return check that consults
`is_demo()` and either errors or swaps in snapshot data.

### 4.3 Config addition

```python
# src/utils/config.py
class Settings(BaseSettings):
    ...
    DEMO_MODE: bool = False
    SNAPSHOT_DIR: str = "data/snapshot"
```

## 5. The snapshot

### 5.1 Layout

```
data/snapshot/
├── manifest.json
├── firms/
│   ├── viirs_mainland.geojson
│   ├── viirs_madeira.geojson
│   ├── viirs_azores.geojson
│   ├── modis_mainland.geojson
│   ├── modis_madeira.geojson
│   └── modis_azores.geojson
├── sentinel2/
│   ├── manifest.json
│   └── thumbnails/*.jpg
├── sentinel3/
│   ├── scenes.geojson
│   └── fire_pixels.geojson
├── eumetsat_frp_sample.png
├── tiles/
│   ├── tile_001.npy ... tile_010.npy
│   ├── tile_001.png ... tile_010.png
│   └── predictions.json
├── fires.sqlite                     # historical timeline
└── aois.json                        # 1–2 read-only demo AOIs
```

`manifest.json` example:

```json
{
  "frozen_at": "2026-05-06T10:00:00Z",
  "snapshot_range": {"start": "2024-08-12", "end": "2024-08-19"},
  "regions": ["mainland", "madeira", "azores"],
  "fire_count": {"viirs": 3421, "modis": 812, "sentinel3": 47},
  "tile_count": 10,
  "model_version": "best_model.pth",
  "model_test_metrics": {"f1": 0.9847, "auc": 0.9955}
}
```

### 5.2 How it's built

`ml/scripts/build_snapshot.py` — runs on Afonso's laptop, with
`DEMO_MODE=false`:

1. Pick a fire-rich window (CLI arg, default 2024-08-12 → 2024-08-19).
2. Hit FIRMS once per (product × region) → save GeoJSON.
3. STAC search Sentinel-2 for that window, low cloud → grab 6–10
   thumbnails; rewrite `assets.thumbnail.href` to point to a local
   `/static/snapshot/sentinel2/...` path.
4. STAC search Sentinel-3 SLSTR → save fire-pixel GeoJSON.
5. Save one EUMETSAT FRP PNG capture as fallback.
6. Pick 10 representative tiles from `ml/data/processed/splits/` (5
   true-positive, 3 true-negative, 1 false-positive, 1
   false-negative) → copy `.npy`, render PNG, run model, store
   `predictions.json` with `{tile_id, lat, lon, fire_probability,
   y_true, outcome, scene_date, cloud_cover}`.
7. Build `fires.sqlite` for the timeline UI. Two sources, in
   preference order: (a) dump from local Postgres if it has 12+
   months of data, (b) fall back to looping the FIRMS API across
   the past year in 7-day windows and inserting the rows. Schema
   mirrors `fire_events` (id, source, detected_at, lat, lon, frp,
   confidence, region) with a single `idx_detected_at` index.
8. Write `manifest.json`.

### 5.3 Size budget

| Artefact | Approx size |
|---|---|
| FIRMS GeoJSONs (6 files) | 1–3 MB |
| S2 thumbnails (8 × 200 KB) | ~2 MB |
| S2 manifest | <50 KB |
| S3 GeoJSONs | <500 KB |
| EUMETSAT PNG | <500 KB |
| Tiles (`.npy` 256 KB × 10 + PNG 50 KB × 10) | ~3 MB |
| `predictions.json` | <100 KB |
| `fires.sqlite` (~200k points, 1 yr) | 10–25 MB |
| `aois.json` | <10 KB |
| **Total committed** | **~20–35 MB** |

Use Git LFS for `.npy`, `.sqlite`, and PNG/JPG above 100 KB. The rest
goes in plain Git.

### 5.4 Refresh cadence

Manual. Run the script on your laptop whenever you want a fresher
demo (every few months for a portfolio piece is plenty), commit the
new `data/snapshot/`, push to GitHub, then `git push hf main` to
redeploy the Space.

## 6. Hard limits (defence in depth)

| Layer | Mechanism | Setting |
|---|---|---|
| App | `slowapi` per-IP | already 30/min on FIRMS, 20/min on inference; keep |
| App | Response size caps | history `LIMIT 5000`; same for any future CSV export |
| App | Demo-mode short-circuit | sensitive routes return 403 or snapshot |
| App | Disabled scheduler | `SCHEDULER_ENABLED` ignored when `DEMO_MODE=true` |
| App | `Cache-Control` headers | `/static/*` 1 day, `/api/firms/*` 5 min, `/api/inference/metrics` 1 day — cached by browsers + HF's reverse proxy |
| HF Spaces | Auto-sleep | Free Spaces sleep after ~48 h of no traffic; wake on next request (~30 s) |
| HF Spaces | Hardware cap | 2 vCPU · 16 GB RAM (Free CPU basic). Demo idles at ~150 MB so we have ~100× headroom |
| HF Spaces | Bandwidth | No documented hard cap on free; HF reserves the right to throttle abusive Spaces. Demo responses are tiny + cached, so this is effectively unreachable |
| HF Spaces | Auto-restart | Spaces auto-restart on crash; no manual intervention needed |

## 7. Secret handling

1. **No secrets in the public image.** The Dockerfile sets
   `ENV DEMO_MODE=true` as the default. All other env vars are
   optional — code paths that need them are unreachable in demo
   mode.
2. **`.env`** stays gitignored as today. Add **`.env.production`**
   (also gitignored) for any future deploy that flips
   `DEMO_MODE=false`. Document in `README.md` that secrets go
   through `fly secrets set` only, never into the image.
3. **`.dockerignore`** explicitly excludes:
   - `.env`, `.env.*`
   - `data/ml/` (the 38 GB training corpus)
   - `ml/training/runs/` (TensorBoard)
   - `ml/training/checkpoints/best_model.pth` (only copied if
     `INCLUDE_MODEL=true` build arg, used for laptop image)
   - `*.sqlite-journal`, `*.db-journal`
   - `tests/`, `docs/`, `*.md`
4. **Multi-stage Dockerfile** — final stage is `python:3.12-slim`
   plus `gdal`, copying only `src/`, `ml/inference/`, `ml/models/`,
   `data/snapshot/`, and (conditionally) `best_model.pth`. Final
   image target ≤ 400 MB.
5. **Path-traversal patch** in `src/api/images.py` — must land before
   any public deploy. Same hardening for any new endpoint that takes
   a filename (snapshot reads use whitelisted filenames only).
6. **CI (`.github/workflows/ci.yml`)** never receives any of these
   secrets. The deploy step is a manual `git push hf main`; no
   GitHub Action and no CI secrets are required for the demo
   pipeline.

## 8. Frontend changes

- On page load, `app.js` calls `/api/health`. If `demo === true`:
  - Hide `dataset.html` / `inference.html` admin links *that imply
    write access* (Model Eval read-only stays visible).
  - Show a sticky info banner: *"Demo mode — frozen snapshot of
    Portugal fires from 12–19 August 2024. [What's this?](#about)"*
  - Disable the AOI "Save" + "Subscribe" buttons with a tooltip
    explaining they're disabled in demo.
  - When the user picks a date outside the snapshot range, show a
    soft warning ("This demo only covers Aug 12–19, 2024 — showing
    snapshot range instead").
- The "Refresh Fire Data" button still works but always returns the
  same snapshot — fine, no special handling needed.
- ML inference page: hide the live-fetch toggle; predictions tab
  shows the curated 10-tile grid.

## 9. Edge / DNS setup

The HF-assigned `amteodoro-portugal-fires.hf.space` hostname comes with
HTTPS and a valid cert managed by Hugging Face. No DNS records to
configure.

If a registered domain is added later, HF Spaces Pro plan supports
custom domains (or you can put Cloudflare in front).

In-app HTTP caching headers help the browser/CDN reuse responses:

```python
# src/main.py — add a small middleware in demo mode
@app.middleware("http")
async def cache_headers(request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "public, max-age=86400"
    elif request.url.path.startswith("/api/firms/"):
        response.headers["Cache-Control"] = "public, max-age=300"
    elif request.url.path.startswith("/api/inference/metrics"):
        response.headers["Cache-Control"] = "public, max-age=86400"
    return response
```

HF Spaces puts a reverse proxy in front of every Space; combined with
browser cache, this absorbs most repeat traffic.

## 10. Hugging Face Spaces setup

HF Spaces uses YAML frontmatter at the top of `README.md` to configure
the Space. We add a Spaces-specific `README.md` (or replace the project
README with a Spaces-aware one) carrying:

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
short_description: Multi-source satellite fire detection over Portugal
---
```

`sdk: docker` tells HF to build and run our existing `Dockerfile`. The
container must listen on the port declared in `app_port`.

### 10.1 Connecting the GitHub repo to HF Spaces

Two practical options:

1. **Mirror push** (simplest, recommended): create the Space at
   `https://huggingface.co/spaces/amteodoro/portugal-fires`, then add
   an HF git remote locally and push to it whenever you want to
   redeploy:

   ```bash
   git remote add hf https://huggingface.co/spaces/amteodoro/portugal-fires
   git push hf main
   ```

2. **GitHub Action** (if you want auto-deploy on every merge): use
   `huggingface/huggingface_hub` action to push from CI.

Option 1 keeps deploys explicit, which fits the set-and-forget
profile.

### 10.2 Secrets

HF Spaces has a secrets UI at *Settings → Variables and secrets*.
Public Spaces only ever need `DEMO_MODE=true` declared as a *Variable*.
No secrets are required because demo-mode code paths never call any
external API. If we ever flip the public Space to non-demo, we'd add
`FIRMS_API_KEY`, `DATABASE_URL`, etc. as proper *Secrets* (encrypted at
rest, never visible after creation, never bundled into the image).

### 10.3 Hardware tier

- **Free CPU basic**: 2 vCPU · 16 GB RAM, sleeps after ~48 h.
- Demo mode (no model load, snapshot reads only) idles at ~150 MB —
  comfortable.
- Switching to non-demo on the same hardware fits easily even with the
  PyTorch model loaded (~600 MB peak).

## 11. Step-by-step implementation plan

Each step ends with a commit on `feat/public-demo`.

1. **Branch + security debt**
   - `git checkout -b feat/public-demo`
   - Patch `src/api/images.py`: resolve path, verify it stays under
     `/app/data/`; whitelist `samples/`, `snapshot/` folders only.
   - Replace `assert` in `src/utils/geo.py` with `ValueError`
     (already may be done — confirm).
   - Replace `print()` in `src/services/firms_service.py` with
     `logger.info`, scrub the API key from any URL log.
   - Add tests for each.

2. **Config + demo-mode plumbing**
   - Add `DEMO_MODE` and `SNAPSHOT_DIR` to `src/utils/config.py`.
   - Create `src/utils/demo.py` (`demo_blocked`, `is_demo`).
   - Wire `main.py` to skip `dataset_*` routers when demo.
   - Add `demo` field to `/api/health`.

3. **Snapshot reader services**
   - `src/services/snapshot.py` with `read_firms`, `read_history`,
     `read_sentinel2`, `read_sentinel3`, `read_eumetsat_frp`,
     `read_aois`, `read_tile`, `read_predictions`, `read_manifest`.
   - Update `firms_service.fetch_bbox`,
     `sentinel_service.search_items`,
     `sentinel3.*`, `eumetsat_service.fetch_frp_png`,
     `aoi.list_aois`, `inference.predict`/`tile_image`/`metrics`/
     `test_results`/`roc_data`/`model_info` to delegate to
     snapshot when `is_demo()`.
   - In demo mode the FastAPI lifespan **does not load the
     PyTorch model**; `inference.predict` short-circuits to a
     `predictions.json` lookup (nearest snapshot tile by
     `tile_id` or by lat/lon).
   - Block `aoi.create_aoi`, `aoi.delete_aoi`,
     `aoi.subscribe_aoi`, `aoi.unsubscribe_aoi` with
     `demo_blocked`.

4. **History via SQLite**
   - Add `aiosqlite` dependency.
   - `src/services/snapshot.py::read_history` runs the same
     bbox/date/source filter against `data/snapshot/fires.sqlite`.
   - `src/api/firms.py::history` switches on `is_demo()`.

5. **Snapshot builder**
   - `ml/scripts/build_snapshot.py` (CLI, `argparse`).
   - Writes `data/snapshot/` deterministically.
   - Document in `README.md` how to refresh.

6. **Generate the first snapshot**
   - On laptop with full creds: `uv run python
     ml/scripts/build_snapshot.py --start 2024-08-12 --end
     2024-08-19`.
   - `git lfs install`, track `*.npy`, `*.sqlite`, `*.jpg`, `*.png`
     under `data/snapshot/`.
   - Commit.

7. **Frontend gating**
   - Edit `src/static/js/app.js` and `inference.js` (if exists) to
     fetch `/api/health` first, store `window.__DEMO__`.
   - Hide write controls; add demo banner.
   - Make sure error messages are friendly when demo blocks
     something the user attempted.

8. **Image hygiene**
   - Update `.dockerignore` (see §7).
   - Convert `Dockerfile` to multi-stage: builder stage installs
     deps, runtime stage copies `src/`, `ml/inference/`,
     `ml/models/`, `data/snapshot/`, optional model checkpoint.
   - Build arg `INCLUDE_MODEL` (default `false` in production,
     `true` for laptop).

9. **Tests**
   - `tests/test_demo_mode.py`:
     - `/api/health` reports `demo: true`.
     - Every gated `POST` returns 403.
     - `/api/firms/viirs` returns the snapshot file content.
     - `/api/firms/history` returns rows from SQLite.
     - `/api/inference/predict` with `live=True` runs in non-live
       path and returns a tile from the snapshot.
   - Existing 62 tests keep passing in both modes.

10. **Hugging Face Spaces wiring**
    - Create a new Space at
      `https://huggingface.co/new-space` named
      `amteodoro/portugal-fires`, SDK = Docker, hardware = CPU
      basic (free).
    - Add HF README frontmatter (§10) at the top of `README.md`.
    - Add HF git remote: `git remote add hf
      https://huggingface.co/spaces/amteodoro/portugal-fires`.
    - In the Space's Settings UI, add Variable
      `DEMO_MODE=true`.
    - First deploy: `git push hf main` (HF will pull LFS objects
      from GitHub LFS — see Task 16 in the plan).
    - Verify `https://amteodoro-portugal-fires.hf.space/api/health`
      returns `demo:true`.

11. **CI**
    - Existing workflow already runs pytest on PRs. Add a job that
      runs `docker build -t test-image .` to catch
      `.dockerignore`/Dockerfile breakage.
    - Optionally a `fly deploy` job on push to `main` using
      the `huggingface_hub` action and `HF_TOKEN` secret. Optional
      for v1 — the plan uses manual `git push hf main` instead.

12. **Personal-site link**
    - Update `amteodoro.github.io` (your GitHub Pages user site)
      to link to `https://amteodoro-portugal-fires.hf.space` with a one-line
      description, e.g. *"Portugal Fire Detection — live satellite
      fire-mapping demo (FastAPI + PyTorch + Leaflet)"*.

13. **Merge to `main`** and deploy.

## 12. Risks & mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Visitor floods `/api/inference/predict` | Medium | Tile read is a 256 KB `.npy` + cached prediction lookup — costs nothing. CF rate-limit handles abuse. |
| GitHub LFS bandwidth (1 GB/mo free) exceeded | Low | The HF mirror keeps its own LFS copy after the first push. Subsequent HF rebuilds pull from HF's LFS, not GitHub's, so the GitHub LFS quota only matters during initial onboarding (~50 MB total). |
| HF Space OOM | Very low | Free CPU basic gives 16 GB RAM; demo mode idles at ~150 MB. |
| HF Space sleep frustration | Low | 48 h sleep window means most visits land warm. Cold-start ~30 s with a clear loading state in the UI. |
| Path traversal exploit | Was open | Patched in step 1, verified with test. |
| Stale snapshot frustrates visitors | Low | Banner explains the date range. Refresh cadence documented. |
| Someone forks the repo and deploys with `DEMO_MODE=false` using their own keys | n/a | That's fine — they're using their own keys. |
| Accidentally committing `.env` or 38 GB ml dir | High if not careful | `.dockerignore` + `.gitignore` audit + a CI guard that fails the build if any committed file matches `data/ml/**` or `*.env`. |

## 13. Open questions

None at design time. All four scoping questions answered:
free-tier hosting, public scope = map+history+ML eval, default
demo (env-flippable), set-and-forget.

## 14. Appendix — endpoint inventory

(Confirms no surface area was missed.)

```
GET    /api/health                         always
GET    /api/config                         always
GET    /api/firms/viirs                    snapshot in demo
GET    /api/firms/modis                    snapshot in demo
GET    /api/firms/history                  sqlite in demo
GET    /api/eumetsat/frp/latest            png in demo
GET    /api/sentinel2/search               manifest in demo
GET    /api/sentinel2/thumbnail            manifest in demo
GET    /api/sentinel3/scenes               geojson in demo
GET    /api/sentinel3/fire-pixels          geojson in demo
GET    /api/images/{folder}/{filename}     hardened, snapshot+samples only
GET    /api/inference/metrics              snapshot in demo
GET    /api/inference/test-results         snapshot in demo
GET    /api/inference/model-info           snapshot in demo
GET    /api/inference/status               always (reports loaded=false in demo)
POST   /api/inference/predict              live forced false in demo
GET    /api/inference/tile-image/{id}      snapshot tiles in demo
GET    /api/inference/roc-data             snapshot in demo
GET    /api/aoi                            snapshot demo AOIs
POST   /api/aoi                            403 in demo
DELETE /api/aoi/{id}                       403 in demo
GET    /api/aoi/{id}/fires                 snapshot in demo
POST   /api/aoi/{id}/subscribe             403 in demo
DELETE /api/aoi/{id}/subscribe             403 in demo
*      /api/dataset/*                      not registered in demo
```
