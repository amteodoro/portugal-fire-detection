"""
Dataset summary API endpoints.

Provides summary statistics, quick stats, and cache management for locally
available training data.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from pathlib import Path
import json
import sys
import shutil
from typing import Dict, Optional

router = APIRouter()

# Add ml directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ---------------------------------------------------------------------------
# Summary endpoints
# ---------------------------------------------------------------------------

@router.get("/summary")
async def get_dataset_summary() -> Dict:
    """
    Get dataset summary with statistics and sample images.

    Returns:
        Dictionary with Sentinel-2, FIRMS, and processed tile statistics
    """
    summary_path = Path("data/dataset_summary.json")

    # Check if summary exists
    if not summary_path.exists():
        # Try to generate it
        try:
            from ml.data.summarize_dataset import generate_summary
            summary = generate_summary()
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate dataset summary: {str(e)}"
            )
    else:
        # Load existing summary
        with open(summary_path, 'r') as f:
            summary = json.load(f)

        # Validate: if summary claims tiles exist but the actual directories are empty,
        # regenerate so the UI doesn't show stale "READY" status.
        claimed_tiles = summary.get("processed", {}).get("total_tiles", 0)
        if claimed_tiles > 0:
            splits_dir = Path("ml/data/processed/splits")
            actual_tiles = sum(
                1 for f in splits_dir.rglob("*.npy") if f.exists()
            ) if splits_dir.exists() else 0
            if actual_tiles == 0:
                try:
                    from ml.data.summarize_dataset import generate_summary
                    summary = generate_summary()
                except Exception:
                    pass  # Return stale summary rather than failing

    return summary


@router.post("/summary/refresh")
async def refresh_dataset_summary() -> Dict:
    """
    Regenerate dataset summary (scans all data directories).

    Returns:
        Updated dataset summary
    """
    try:
        from ml.data.summarize_dataset import generate_summary
        summary = generate_summary()
        return summary
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to refresh dataset summary: {str(e)}"
        )


@router.get("/stats")
async def get_quick_stats() -> Dict:
    """
    Get quick dataset statistics without generating images.

    Returns:
        Basic dataset counts
    """
    summary_path = Path("data/dataset_summary.json")

    if not summary_path.exists():
        return {
            "sentinel2_scenes": 0,
            "fire_detections": 0,
            "training_tiles": 0,
            "ready_for_training": False
        }

    with open(summary_path, 'r') as f:
        summary = json.load(f)

    return {
        "sentinel2_scenes": summary['sentinel2']['total_scenes'],
        "fire_detections": summary['firms']['total_fire_detections'],
        "training_tiles": summary['processed']['total_tiles'],
        "fire_tiles": summary['processed']['fire_tiles'],
        "no_fire_tiles": summary['processed']['no_fire_tiles'],
        "ready_for_training": summary['status']['ready_for_training']
    }


# ---------------------------------------------------------------------------
# Cache management helpers
# ---------------------------------------------------------------------------

class CacheStatsRequest(BaseModel):
    cache_type: str  # 'sentinel2', 'sentinel3', 'firms', 'tiles', 'all'


class CacheDeleteRequest(BaseModel):
    cache_type: str
    confirm: bool = False


def get_directory_size(path: Path) -> int:
    """Calculate total size of directory in bytes (not following symlinks)."""
    total = 0
    try:
        for item in path.rglob('*'):
            # Use lstat to not follow symlinks (avoid double-counting)
            stat_info = item.lstat()
            if item.is_file() and not item.is_symlink():
                total += stat_info.st_size
    except Exception:
        pass
    return total


def count_files(path: Path, pattern: str = '*') -> int:
    """Count regular files (not symlinks) matching pattern in directory."""
    try:
        return len([f for f in path.rglob(pattern) if f.is_file() and not f.is_symlink()])
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Cache endpoints
# ---------------------------------------------------------------------------

@router.get("/cache/stats")
async def get_cache_stats(cache_type: Optional[str] = None) -> Dict:
    """
    Get statistics about cached data.

    Args:
        cache_type: Optional filter ('sentinel2', 'sentinel3', 'firms', 'tiles', 'samples')

    Returns:
        Dictionary with cache statistics
    """
    stats = {}

    # Define cache directories (relative to project root)
    # Each should be mutually exclusive to avoid double-counting
    cache_dirs = {
        'sentinel2': Path('data/ml/sentinel2'),      # Raw S2 imagery
        'sentinel3': Path('data/ml/sentinel3'),      # Raw S3 thermal data
        'firms': Path('data/firms'),                  # FIRMS fire detections
        'tiles': Path('ml/data/tiles'),               # Extracted training tiles
        'labels': Path('ml/data/labels'),             # Tile labels
        'samples': Path('data/samples'),              # Visualization samples
        'processed': Path('ml/data/processed')        # CSV/JSON metadata only
    }

    # If specific cache_type requested, filter
    if cache_type and cache_type in cache_dirs:
        cache_dirs = {cache_type: cache_dirs[cache_type]}

    for name, path in cache_dirs.items():
        if not path.exists():
            stats[name] = {
                'exists': False,
                'size_bytes': 0,
                'size_mb': 0,
                'file_count': 0,
                'path': str(path)
            }
            continue

        size_bytes = get_directory_size(path)
        file_count = count_files(path)

        stats[name] = {
            'exists': True,
            'size_bytes': size_bytes,
            'size_mb': round(size_bytes / (1024 * 1024), 2),
            'size_gb': round(size_bytes / (1024 * 1024 * 1024), 2),
            'file_count': file_count,
            'path': str(path),
            'deletable': True  # All are safe to delete (not code)
        }

    # Calculate total
    total_size = sum(s['size_bytes'] for s in stats.values())
    total_files = sum(s['file_count'] for s in stats.values())

    return {
        'caches': stats,
        'total_size_bytes': total_size,
        'total_size_mb': round(total_size / (1024 * 1024), 2),
        'total_size_gb': round(total_size / (1024 * 1024 * 1024), 2),
        'total_files': total_files
    }


@router.post("/cache/delete")
async def delete_cache(request: CacheDeleteRequest) -> Dict:
    """
    Delete cached data with safety checks.

    Args:
        request: Delete request with cache_type and confirmation

    Returns:
        Result of deletion operation
    """
    cache_type = request.cache_type

    # Define safe deletion targets
    safe_cache_dirs = {
        'sentinel2': Path('data/ml/sentinel2'),
        'sentinel3': Path('data/ml/sentinel3'),
        'firms': Path('data/firms'),
        'tiles': Path('ml/data/tiles'),
        'labels': Path('ml/data/labels'),
        'samples': Path('data/samples'),
        'processed': Path('ml/data/processed'),
        'all': None  # Special case
    }

    if cache_type not in safe_cache_dirs:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid cache_type: {cache_type}. Must be one of: {list(safe_cache_dirs.keys())}"
        )

    # Get stats before deletion
    stats_before = await get_cache_stats(cache_type if cache_type != 'all' else None)

    if not request.confirm:
        # Return preview of what will be deleted
        if cache_type == 'all':
            preview = {
                'action': 'preview',
                'cache_type': cache_type,
                'warning': 'This will delete ALL cached data!',
                'items_to_delete': stats_before['caches'],
                'total_size_mb': stats_before['total_size_mb'],
                'total_files': stats_before['total_files'],
                'require_confirmation': True
            }
        else:
            cache_stats = stats_before['caches'].get(cache_type, {})
            preview = {
                'action': 'preview',
                'cache_type': cache_type,
                'warning': f'This will delete all {cache_type} data',
                'size_mb': cache_stats.get('size_mb', 0),
                'file_count': cache_stats.get('file_count', 0),
                'path': cache_stats.get('path', ''),
                'require_confirmation': True
            }

        return preview

    # User confirmed - proceed with deletion
    deleted_items = []
    errors = []

    if cache_type == 'all':
        # Delete all caches
        for name, path in safe_cache_dirs.items():
            if name == 'all' or path is None:
                continue

            if path.exists():
                try:
                    shutil.rmtree(path)
                    deleted_items.append({
                        'name': name,
                        'path': str(path),
                        'status': 'deleted'
                    })
                except Exception as e:
                    errors.append({
                        'name': name,
                        'path': str(path),
                        'error': str(e)
                    })
    else:
        # Delete specific cache
        path = safe_cache_dirs[cache_type]

        if path.exists():
            try:
                shutil.rmtree(path)
                deleted_items.append({
                    'name': cache_type,
                    'path': str(path),
                    'status': 'deleted'
                })
            except Exception as e:
                errors.append({
                    'name': cache_type,
                    'path': str(path),
                    'error': str(e)
                })
        else:
            return {
                'action': 'delete',
                'cache_type': cache_type,
                'status': 'not_found',
                'message': f'Cache directory does not exist: {path}'
            }

    # Get stats after deletion
    stats_after = await get_cache_stats(cache_type if cache_type != 'all' else None)

    return {
        'action': 'delete',
        'cache_type': cache_type,
        'status': 'success' if not errors else 'partial',
        'deleted': deleted_items,
        'errors': errors,
        'stats_before': stats_before,
        'stats_after': stats_after,
        'space_freed_mb': stats_before['total_size_mb'] - stats_after['total_size_mb']
    }
