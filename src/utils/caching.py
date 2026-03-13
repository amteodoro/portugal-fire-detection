import json
import time
import os
from cachetools import TTLCache

# Per-type memory caches with TTLs matched to each data source's freshness
firms_cache = TTLCache(maxsize=64, ttl=900)      # matches FIRMS_TTL
eumetsat_cache = TTLCache(maxsize=32, ttl=1800)  # matches EUMETSAT_TTL
s2_cache = TTLCache(maxsize=32, ttl=86400)       # matches S2_TTL

_CACHES = {
    "firms": firms_cache,
    "eumetsat": eumetsat_cache,
    "s2": s2_cache,
}


def cache_get(key: str, cache_type: str = "firms"):
    return _CACHES[cache_type].get(key)


def cache_set(key: str, value, ttl: int | None = None, cache_type: str = "firms"):
    # TTLCache.ttl is read-only; all items in a cache share the cache's global TTL.
    # The ttl parameter is accepted for API compatibility but ignored.
    _CACHES[cache_type][key] = value


def file_cache_path(dir_path: str, key: str) -> str:
    safe = key.replace("/", "_").replace(":", "_").replace("?", "_")
    return os.path.join(dir_path, f"{safe}.json")


def file_cache_load(dir_path: str, key: str, max_age_sec: int) -> dict | None:
    path = file_cache_path(dir_path, key)
    if not os.path.exists(path):
        return None
    if (time.time() - os.path.getmtime(path)) > max_age_sec:
        return None
    with open(path, "r") as f:
        return json.load(f)


def file_cache_store(dir_path: str, key: str, data: dict):
    os.makedirs(dir_path, exist_ok=True)
    with open(file_cache_path(dir_path, key), "w") as f:
        json.dump(data, f)
