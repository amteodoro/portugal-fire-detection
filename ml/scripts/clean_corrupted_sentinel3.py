#!/usr/bin/env python3
"""
Clean up corrupted Sentinel-3 downloads.

Finds and removes incomplete/corrupted .zip files so they can be re-downloaded.
"""

import zipfile
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_zip_integrity(zip_path: Path) -> bool:
    """
    Check if a zip file is complete and valid.
    
    Args:
        zip_path: Path to .zip file
        
    Returns:
        True if valid, False if corrupted
    """
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            # Try to read the file list (requires end-of-central-directory)
            files = zf.namelist()
            return True
    except zipfile.BadZipFile:
        return False
    except Exception as e:
        logger.warning(f"Error checking {zip_path.name}: {e}")
        return False


def find_corrupted_zips(data_dir: Path = Path("data/ml/sentinel3")) -> list[Path]:
    """
    Find all corrupted .zip files in Sentinel-3 data directory.
    
    Args:
        data_dir: Root directory to search
        
    Returns:
        List of corrupted .zip file paths
    """
    corrupted = []
    
    zip_files = list(data_dir.rglob("*.zip"))
    logger.info(f"Checking {len(zip_files)} .zip files for corruption...")
    
    for zip_path in zip_files:
        if not check_zip_integrity(zip_path):
            corrupted.append(zip_path)
    logger.info(f"Found {len(corrupted)} corrupted files")
    
    return corrupted


def clean_corrupted_files(dry_run: bool = True):
    """
    Find and optionally remove corrupted Sentinel-3 .zip files.
    
    Args:
        dry_run: If True, only list corrupted files. If False, delete them.
    """
    corrupted = find_corrupted_zips()
    
    if not corrupted:
        logger.info("✓ No corrupted files found!")
        return
    
    logger.info(f"\nCorrupted files:")
    total_size = 0
    for path in corrupted:
        size_mb = path.stat().st_size / (1024 * 1024)
        total_size += size_mb
        logger.info(f"  - {path} ({size_mb:.1f} MB)")
    
    logger.info(f"\nTotal corrupted data: {total_size:.1f} MB")
    
    if dry_run:
        logger.info("\n⚠️  DRY RUN - No files will be deleted")
        logger.info("Re-run with --delete flag to actually remove files")
    else:
        logger.info("\n🗑️  Deleting corrupted files...")
        for path in corrupted:
            try:
                path.unlink()
                logger.info(f"  ✓ Deleted {path.name}")
            except Exception as e:
                logger.error(f"  ✗ Failed to delete {path.name}: {e}")
        
        logger.info(f"\n✅ Cleaned up {len(corrupted)} corrupted files")
        logger.info("Re-run the Sentinel-3 download script to re-download them")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Clean up corrupted Sentinel-3 downloads")
    parser.add_argument('--delete', action='store_true', 
                       help='Actually delete corrupted files (default: dry run)')
    parser.add_argument('--data-dir', default='data/ml/sentinel3',
                       help='Sentinel-3 data directory')
    
    args = parser.parse_args()
    
    clean_corrupted_files(dry_run=not args.delete)
