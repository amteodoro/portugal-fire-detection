import json
import time
import os
from cachetools import TTLCache

memory_cache = TTLCache(maxsize=128, ttl=600)

def cache_get(key: str):
    return memory_cache.get(key)

def cache_set(key: str, value, ttl: int | None = None):
    if ttl:
        memory_cache.ttl = ttl
    memory_cache[key] = value

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
