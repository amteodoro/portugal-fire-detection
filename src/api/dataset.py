"""
Dataset summary API endpoints.

Provides information about locally available training data.
"""

from fastapi import APIRouter, HTTPException
from pathlib import Path
import json
import sys
from typing import Dict

router = APIRouter()

# Add ml directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


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
