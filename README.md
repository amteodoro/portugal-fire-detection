# Portugal Fire Detection Web Application

Multi-source satellite fire detection viewer for Portugal using VIIRS/MODIS NRT feeds (FIRMS), Meteosat FRP products (EUMETSAT), and Sentinel-2 for post-fire damage mapping.

## Features

- **VIIRS/MODIS Active Fire Detection** via NASA FIRMS API (country-scoped to Portugal)
- **Meteosat FRP Continuous Monitoring** via EUMETSAT WMS overlay
- **Sentinel-2 Post-Fire Damage Mapping** via STAC catalog search
- **Interactive Leaflet Map** centered on Portugal
- **Aggressive Caching** to minimize API usage during testing
- **Docker-Based Environment** with uv package management

## Prerequisites

- Docker
- Docker Compose
- API Keys (see below)

## API Key Registration

### Required:
- **NASA FIRMS**: Register at https://firms.modaps.eosdis.nasa.gov/api/ to get your API key

### Optional (for enhanced features):
- **EUMETSAT**: Register at https://eoportal.eumetsat.int/ for WMS token (public WMS available)
- **Copernicus Data Space**: Register at https://dataspace.copernicus.eu/ (defaults to public Earth Search STAC)

## Setup

1. **Clone the repository** (if not already in it):
   ```bash
   cd portugal-fire-detection
   ```

2. **Create `.env` file from template**:
   ```bash
   cp .env.example .env
   ```

3. **Edit `.env` and add your API keys**:
   ```bash
   # At minimum, add your FIRMS API key:
   FIRMS_API_KEY=your_actual_api_key_here
   ```

4. **Build and run the Docker container**:
   ```bash
   docker-compose up --build
   ```

5. **Open the web application**:
   Navigate to http://localhost:8000

## Usage

### Web Interface

- **VIIRS Layer**: Toggle to show/hide VIIRS fire detections (checked by default)
- **MODIS Layer**: Toggle to show/hide MODIS fire detections
- **Meteosat FRP Layer**: Toggle to show/hide FRP overlay image
- **Sentinel-2 Search**: Configure days and cloud cover percentage, then click "Search" to fetch thumbnails

### API Endpoints

All endpoints return JSON (except images):

- `GET /api/health` - Health check
- `GET /api/firms/viirs?days=1` - VIIRS fire detections (1-7 days)
- `GET /api/firms/modis?days=1` - MODIS fire detections (1-7 days)
- `GET /api/eumetsat/frp/latest` - Latest Meteosat FRP image metadata
- `GET /api/sentinel2/search?days=7&cloud_pct=20&limit=12` - Sentinel-2 STAC search
- `GET /api/sentinel2/thumbnail?days=7&cloud_pct=20` - Latest Sentinel-2 thumbnail
- `GET /api/images/{folder}/{filename}` - Serve cached images

## Configuration

### Geographic Scope

The `PORTUGAL_BBOX` in `.env` defines the bounding box:
```
PORTUGAL_BBOX=-9.5,36.8,-6.0,42.3  # Mainland Portugal
```

To include Madeira and Açores, adjust coordinates accordingly.

### Background Scheduler

By default, `SCHEDULER_ENABLED=false` to minimize API usage during testing. When enabled:
- FIRMS: Fetches every 30 minutes
- EUMETSAT FRP: Fetches every 1 hour
- Sentinel-2: Searches daily at 02:00 UTC
- Cleanup: Runs daily at 03:00 UTC (14-day retention)

### Cache TTLs

Adjust cache durations in `.env`:
- `FIRMS_TTL=900` (15 minutes)
- `EUMETSAT_TTL=1800` (30 minutes)
- `S2_TTL=86400` (24 hours)

## Development

### Running Tests

**IMPORTANT**: All tests must be executed inside the Docker container:

```bash
docker-compose exec web uv run pytest
```

### Installing New Dependencies

Use `uv` only (never pip) inside the container:

```bash
docker-compose exec web uv add <package-name>
```

### File Structure

```
portugal-fire-detection/
├── src/
│   ├── main.py                 # FastAPI application entry
│   ├── api/                    # API route handlers
│   │   ├── firms.py
│   │   ├── eumetsat.py
│   │   ├── sentinel.py
│   │   └── images.py
│   ├── services/               # Business logic & external API calls
│   │   ├── firms_service.py
│   │   ├── eumetsat_service.py
│   │   └── sentinel_service.py
│   ├── models/                 # Pydantic models
│   │   ├── fire.py
│   │   ├── imagery.py
│   │   └── geo.py
│   ├── utils/                  # Shared utilities
│   │   ├── config.py
│   │   ├── http.py
│   │   ├── caching.py
│   │   ├── files.py
│   │   ├── geo.py
│   │   └── cleanup.py
│   └── static/                 # Frontend assets
│       ├── index.html
│       ├── css/styles.css
│       └── js/app.js
├── data/                       # Local image storage (gitignored)
│   ├── firms/
│   ├── eumetsat/
│   └── sentinel2/
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── .env
```

## Data Storage

Images and cached data are stored locally in `./data/`:
- `data/firms/` - JSON cache files
- `data/eumetsat/` - FRP PNG overlays
- `data/sentinel2/` - Thumbnail images

Files older than 14 days are automatically cleaned up when the scheduler is enabled.

## Troubleshooting

### No fire detections showing
- Verify your FIRMS API key is correct in `.env`
- Check that Portugal has active fires (VIIRS/MODIS data availability)
- Try increasing the `days` parameter

### EUMETSAT FRP not loading
- The public WMS endpoint may have availability issues
- Check `EUMETSAT_WMS_URL` and `EUMETSAT_FRP_LAYER` configuration
- Consider obtaining a token from EUMETSAT

### Sentinel-2 thumbnails not appearing
- Adjust cloud cover percentage (try higher values like 50%)
- Increase search days (try 14 or 30 days)
- Default Earth Search STAC may have different availability than Copernicus Data Space

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]

# Portugal Fire Detection Web Application

Multi-source satellite fire detection viewer for Portugal using VIIRS/MODIS NRT feeds (FIRMS), Meteosat FRP products (EUMETSAT), and Sentinel-2 for post-fire damage mapping.

## Features

- **VIIRS/MODIS Active Fire Detection** via NASA FIRMS API (country-scoped to Portugal)
- **Meteosat FRP Continuous Monitoring** via EUMETSAT WMS overlay
- **Sentinel-2 Post-Fire Damage Mapping** via STAC catalog search
- **Interactive Leaflet Map** centered on Portugal
- **Aggressive Caching** to minimize API usage during testing
- **Docker-Based Environment** with uv package management

## Prerequisites

- Docker
- Docker Compose
- API Keys (see below)

## API Key Registration

### Required:
- **NASA FIRMS**: Register at https://firms.modaps.eosdis.nasa.gov/api/ to get your API key

### Optional (for enhanced features):
- **EUMETSAT**: Register at https://eoportal.eumetsat.int/ for WMS token (public WMS available)
- **Copernicus Data Space**: Register at https://dataspace.copernicus.eu/ (defaults to public Earth Search STAC)

## Setup

1. **Clone the repository** (if not already in it):
   ```bash
   cd portugal-fire-detection
   ```

2. **Create `.env` file from template**:
   ```bash
   cp .env.example .env
   ```

3. **Edit `.env` and add your API keys**:
   ```bash
   # At minimum, add your FIRMS API key:
   FIRMS_API_KEY=your_actual_api_key_here
   ```

4. **Build and run the Docker container**:
   ```bash
   docker-compose up --build
   ```

5. **Open the web application**:
   Navigate to http://localhost:8000

## Usage

### Web Interface

- **VIIRS Layer**: Toggle to show/hide VIIRS fire detections (checked by default)
- **MODIS Layer**: Toggle to show/hide MODIS fire detections
- **Meteosat FRP Layer**: Toggle to show/hide FRP overlay image
- **Sentinel-2 Search**: Configure days and cloud cover percentage, then click "Search" to fetch thumbnails

### API Endpoints

All endpoints return JSON (except images):

- `GET /api/health` - Health check
- `GET /api/firms/viirs?days=1` - VIIRS fire detections (1-7 days)
- `GET /api/firms/modis?days=1` - MODIS fire detections (1-7 days)
- `GET /api/eumetsat/frp/latest` - Latest Meteosat FRP image metadata
- `GET /api/sentinel2/search?days=7&cloud_pct=20&limit=12` - Sentinel-2 STAC search
- `GET /api/sentinel2/thumbnail?days=7&cloud_pct=20` - Latest Sentinel-2 thumbnail
- `GET /api/images/{folder}/{filename}` - Serve cached images

## Configuration

### Geographic Scope

The `PORTUGAL_BBOX` in `.env` defines the bounding box:
```
PORTUGAL_BBOX=-9.5,36.8,-6.0,42.3  # Mainland Portugal
```

To include Madeira and Açores, adjust coordinates accordingly.

### Background Scheduler

By default, `SCHEDULER_ENABLED=false` to minimize API usage during testing. When enabled:
- FIRMS: Fetches every 30 minutes
- EUMETSAT FRP: Fetches every 1 hour
- Sentinel-2: Searches daily at 02:00 UTC
- Cleanup: Runs daily at 03:00 UTC (14-day retention)

### Cache TTLs

Adjust cache durations in `.env`:
- `FIRMS_TTL=900` (15 minutes)
- `EUMETSAT_TTL=1800` (30 minutes)
- `S2_TTL=86400` (24 hours)

## Development

### Running Tests

**IMPORTANT**: All tests must be executed inside the Docker container:

```bash
docker-compose exec web uv run pytest
```

### Installing New Dependencies

Use `uv` only (never pip) inside the container:

```bash
docker-compose exec web uv add <package-name>
```

### File Structure

```
portugal-fire-detection/
├── src/
│   ├── main.py                 # FastAPI application entry
│   ├── api/                    # API route handlers
│   │   ├── firms.py
│   │   ├── eumetsat.py
│   │   ├── sentinel.py
│   │   └── images.py
│   ├── services/               # Business logic & external API calls
│   │   ├── firms_service.py
│   │   ├── eumetsat_service.py
│   │   └── sentinel_service.py
│   ├── models/                 # Pydantic models
│   │   ├── fire.py
│   │   ├── imagery.py
│   │   └── geo.py
│   ├── utils/                  # Shared utilities
│   │   ├── config.py
│   │   ├── http.py
│   │   ├── caching.py
│   │   ├── files.py
│   │   ├── geo.py
│   │   └── cleanup.py
│   └── static/                 # Frontend assets
│       ├── index.html
│       ├── css/styles.css
│       └── js/app.js
├── data/                       # Local image storage (gitignored)
│   ├── firms/
│   ├── eumetsat/
│   └── sentinel2/
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── .env
```

## Data Storage

Images and cached data are stored locally in `./data/`:
- `data/firms/` - JSON cache files
- `data/eumetsat/` - FRP PNG overlays
- `data/sentinel2/` - Thumbnail images

Files older than 14 days are automatically cleaned up when the scheduler is enabled.

## Troubleshooting

### No fire detections showing
- Verify your FIRMS API key is correct in `.env`
- Check that Portugal has active fires (VIIRS/MODIS data availability)
- Try increasing the `days` parameter

### EUMETSAT FRP not loading
- The public WMS endpoint may have availability issues
- Check `EUMETSAT_WMS_URL` and `EUMETSAT_FRP_LAYER` configuration
- Consider obtaining a token from EUMETSAT

### Sentinel-2 thumbnails not appearing
- Adjust cloud cover percentage (try higher values like 50%)
- Increase search days (try 14 or 30 days)
- Default Earth Search STAC may have different availability than Copernicus Data Space

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]
