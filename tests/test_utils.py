"""
Unit tests for utility modules.

Run with: docker-compose exec web uv run pytest tests/test_utils.py
"""

import json
import os
import time
from pathlib import Path

import pytest


class TestGeoUtils:
    """Tests for src/utils/geo.py"""

    def test_parse_bbox_valid(self):
        """Test parsing a valid bbox string."""
        from src.utils.geo import parse_bbox

        bbox = parse_bbox("-9.5,36.8,-6.0,42.3")
        assert bbox == (-9.5, 36.8, -6.0, 42.3)

    def test_parse_bbox_with_spaces(self):
        """Test parsing bbox with extra spaces."""
        from src.utils.geo import parse_bbox

        bbox = parse_bbox("-9.5, 36.8, -6.0, 42.3")
        assert bbox == (-9.5, 36.8, -6.0, 42.3)

    def test_parse_bbox_invalid_format(self):
        """Test parsing invalid bbox raises error."""
        from src.utils.geo import parse_bbox

        with pytest.raises(ValueError):
            parse_bbox("-9.5,36.8,-6.0")  # Only 3 values

    def test_parse_bbox_non_numeric(self):
        """Test parsing non-numeric bbox raises error."""
        from src.utils.geo import parse_bbox

        with pytest.raises(ValueError):
            parse_bbox("a,b,c,d")


class TestCachingUtils:
    """Tests for src/utils/caching.py"""

    def test_memory_cache_set_get(self):
        """Test setting and getting from memory cache."""
        from src.utils.caching import cache_get, cache_set

        test_data = {"test": "value", "number": 42}
        cache_set("test_key", test_data)

        result = cache_get("test_key")
        assert result == test_data

    def test_memory_cache_get_nonexistent(self):
        """Test getting nonexistent key returns None."""
        from src.utils.caching import cache_get

        result = cache_get("nonexistent_key_12345")
        assert result is None

    def test_file_cache_path(self):
        """Test file cache path generation."""
        from src.utils.caching import file_cache_path

        path = file_cache_path("/tmp/cache", "test_key")
        assert path == "/tmp/cache/test_key.json"

    def test_file_cache_path_sanitizes_special_chars(self):
        """Test file cache path sanitizes special characters."""
        from src.utils.caching import file_cache_path

        path = file_cache_path("/tmp/cache", "test/key:with?special")
        assert "/" not in path.split("/")[-1].replace(".json", "")
        assert ":" not in path
        assert "?" not in path

    def test_file_cache_store_and_load(self, tmp_path):
        """Test storing and loading from file cache."""
        from src.utils.caching import file_cache_store, file_cache_load

        cache_dir = str(tmp_path / "cache")
        test_data = {"features": [1, 2, 3], "type": "test"}

        file_cache_store(cache_dir, "test_key", test_data)

        # Load with long max age (should succeed)
        result = file_cache_load(cache_dir, "test_key", max_age_sec=3600)
        assert result == test_data

    def test_file_cache_load_expired(self, tmp_path):
        """Test loading expired cache returns None."""
        from src.utils.caching import file_cache_store, file_cache_load

        cache_dir = str(tmp_path / "cache")
        test_data = {"test": "data"}

        file_cache_store(cache_dir, "expired_key", test_data)

        # Wait a tiny bit and load with very short max age
        time.sleep(0.1)
        result = file_cache_load(cache_dir, "expired_key", max_age_sec=0)
        assert result is None

    def test_file_cache_load_nonexistent(self, tmp_path):
        """Test loading nonexistent file returns None."""
        from src.utils.caching import file_cache_load

        cache_dir = str(tmp_path / "cache")
        result = file_cache_load(cache_dir, "nonexistent", max_age_sec=3600)
        assert result is None

    def test_file_cache_creates_directory(self, tmp_path):
        """Test file cache creates directory if not exists."""
        from src.utils.caching import file_cache_store

        cache_dir = str(tmp_path / "new_cache_dir" / "nested")
        test_data = {"test": "data"}

        file_cache_store(cache_dir, "test_key", test_data)

        assert os.path.exists(cache_dir)
        assert os.path.exists(os.path.join(cache_dir, "test_key.json"))


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
