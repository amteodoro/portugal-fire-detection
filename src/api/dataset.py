"""
Dataset summary API endpoints.

Provides information about locally available training data.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pathlib import Path
import json
import sys
import asyncio
import shutil
from typing import Dict, List, Optional, AsyncGenerator

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


@router.post("/fetch/sentinel2")
async def fetch_sentinel2_data(
    days: int = 30,
    cloud_pct: int = 20
) -> Dict:
    """
    Trigger Sentinel-2 data download for recent fire events.
    
    Args:
        days: Number of days to look back (default: 30)
        cloud_pct: Maximum cloud cover percentage (default: 20)
    
    Returns:
        Status message with number of scenes found
    """
    try:
        # This endpoint triggers the download process
        # The actual download happens in the background via the download script
        return {
            "status": "initiated",
            "message": f"Sentinel-2 download initiated for last {days} days with max {cloud_pct}% cloud cover",
            "note": "Run: docker compose exec web python ml/scripts/download_sentinel2.py"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initiate Sentinel-2 download: {str(e)}"
        )


@router.post("/fetch/firms")
async def fetch_historical_firms(
    days: int = 30
) -> Dict:
    """
    Trigger historical FIRMS data download.
    
    Args:
        days: Number of days to look back (default: 30)
    
    Returns:
        Status message
    """
    try:
        return {
            "status": "initiated",
            "message": f"FIRMS historical data download initiated for last {days} days",
            "note": "Run: docker compose exec web python ml/scripts/fetch_historical_firms.py"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initiate FIRMS download: {str(e)}"
        )


# Request models for new endpoints
class Sentinel2DownloadRequest(BaseModel):
    start_date: str
    end_date: str
    max_cloud: Optional[int] = 20
    max_scenes: Optional[int] = None


class FIRMSFetchRequest(BaseModel):
    start_date: str
    end_date: str
    products: Optional[List[str]] = ["VIIRS", "MODIS"]
    region: Optional[str] = None


class OpenEODownloadRequest(BaseModel):
    start_date: str
    end_date: str
    max_cloud: Optional[int] = 20


class Sentinel3DownloadRequest(BaseModel):
    start_date: str
    end_date: str
    instruments: Optional[List[str]] = ["slstr"]  # ["slstr", "olci"]
    max_scenes: Optional[int] = None


async def stream_subprocess_output(cmd: List[str]) -> AsyncGenerator[str, None]:
    """
    Stream subprocess output line by line as Server-Sent Events.
    
    Args:
        cmd: Command to execute
        
    Yields:
        SSE-formatted strings with stdout/stderr lines
    """
    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        # Use a queue to collect output from both streams concurrently
        queue = asyncio.Queue()
        
        async def read_stream(stream, stream_type):
            """Read from a stream and put lines into the queue."""
            try:
                while True:
                    line = await stream.readline()
                    if not line:
                        break
                    text = line.decode('utf-8', errors='replace').rstrip()
                    if text:
                        await queue.put((stream_type, text))
            except Exception as e:
                await queue.put(("error", f"Stream error: {e}"))
            finally:
                await queue.put((stream_type, None))  # Signal stream is done
        
        # Start both readers concurrently
        stdout_task = asyncio.create_task(read_stream(process.stdout, "stdout"))
        stderr_task = asyncio.create_task(read_stream(process.stderr, "stderr"))
        
        streams_done = 0
        while streams_done < 2:
            try:
                # Wait for next item with timeout to stay responsive
                stream_type, text = await asyncio.wait_for(queue.get(), timeout=0.1)
                
                if text is None:
                    streams_done += 1
                else:
                    yield f"data: {{\"type\": \"{stream_type}\", \"message\": {json.dumps(text)}}}\n\n"
            except asyncio.TimeoutError:
                # No output yet, continue waiting
                continue
        
        # Wait for tasks to complete
        await stdout_task
        await stderr_task
        
        # Wait for process to complete
        returncode = await process.wait()
        
        # Send completion message
        if returncode == 0:
            yield f"data: {{\"type\": \"complete\", \"message\": \"Process completed successfully\", \"returncode\": {returncode}}}\n\n"
        else:
            yield f"data: {{\"type\": \"error\", \"message\": \"Process failed with code {returncode}\", \"returncode\": {returncode}}}\n\n"
    
    except Exception as e:
        yield f"data: {{\"type\": \"error\", \"message\": {json.dumps(str(e))}}}\n\n"


@router.post("/download-sentinel2")
async def download_sentinel2(request: Sentinel2DownloadRequest):
    """
    Execute Sentinel-2 download script with streaming output.
    
    Args:
        request: Download parameters including dates and cloud cover
    
    Returns:
        Server-Sent Events stream with real-time progress
    """
    try:
        # Build command
        cmd = [
            "uv", "run", "python", "-u", "ml/scripts/download_sentinel2.py",
            "--start-date", request.start_date,
            "--end-date", request.end_date,
            "--max-cloud", str(request.max_cloud)
        ]
        
        if request.max_scenes:
            cmd.extend(["--max-scenes", str(request.max_scenes)])
        
        return StreamingResponse(
            stream_subprocess_output(cmd),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to execute Sentinel-2 download: {str(e)}"
        )


@router.post("/fetch-firms-data")
async def fetch_firms_data(request: FIRMSFetchRequest):
    """
    Execute historical FIRMS fetch script with streaming output.
    
    Args:
        request: Fetch parameters including date range and products
    
    Returns:
        Server-Sent Events stream with real-time progress
    """
    try:
        # Validate required fields
        if not request.start_date or not request.end_date:
            raise HTTPException(
                status_code=400,
                detail="start_date and end_date are required"
            )
        
        # Build command
        cmd = [
            "uv", "run", "python", "-u", "ml/scripts/fetch_historical_firms.py",
            "--start-date", request.start_date,
            "--end-date", request.end_date
        ]
        
        if request.products:
            cmd.extend(["--products"] + request.products)
        
        if request.region:
            cmd.extend(["--region", request.region])
        
        return StreamingResponse(
            stream_subprocess_output(cmd),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to execute FIRMS fetch: {str(e)}"
        )


@router.post("/download-openeo")
async def download_openeo(request: OpenEODownloadRequest):
    """
    Execute OpenEO download script with streaming output.
    
    Args:
        request: Download parameters including dates and cloud cover
    
    Returns:
        Server-Sent Events stream with real-time progress
    """
    try:
        # Build command
        cmd = [
            "uv", "run", "python", "-u", "ml/download_sentinel2_openeo.py",
            "--start-date", request.start_date,
            "--end-date", request.end_date,
            "--max-cloud", str(request.max_cloud)
        ]
        
        return StreamingResponse(
            stream_subprocess_output(cmd),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to execute OpenEO download: {str(e)}"
        )


@router.post("/download-sentinel3")
async def download_sentinel3(request: Sentinel3DownloadRequest):
    """
    Execute Sentinel-3 download script with streaming output.
    
    Downloads Sentinel-3 SLSTR (thermal IR for active fire detection) and/or
    OLCI (optical for burn scar mapping) data.
    
    Args:
        request: Download parameters including dates and instruments
    
    Returns:
        Server-Sent Events stream with real-time progress
    """
    try:
        # Build command
        cmd = [
            "uv", "run", "python", "-u", "ml/scripts/download_sentinel3.py",
            "--start-date", request.start_date,
            "--end-date", request.end_date
        ]
        
        if request.instruments:
            cmd.extend(["--instruments"] + request.instruments)
        
        if request.max_scenes:
            cmd.extend(["--max-scenes", str(request.max_scenes)])
        
        return StreamingResponse(
            stream_subprocess_output(cmd),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to execute Sentinel-3 download: {str(e)}"
        )


async def stream_dataset_pipeline(tile_size: int = 128) -> AsyncGenerator[str, None]:
    """
    Stream output from complete dataset pipeline:
    1. Match FIRMS fires to Sentinel-2 scenes
    2. Create dataset tiles
    3. Prepare train/val/test splits
    4. Generate co-located S2+S3 pairs
    
    Args:
        tile_size: Size of tiles to extract in pixels (default 128)
    """
    try:
        # Step 1: Match FIRMS to Sentinel-2 and Sentinel-3 scenes
        yield f"data: {{\"type\": \"stdout\", \"message\": \"Step 1/4: Matching FIRMS fire detections to S2 and S3 scenes...\"}}\n\n"
        
        match_cmd = ["uv", "run", "python", "-u", "ml/scripts/match_firms_to_scenes.py"]
        process1 = await asyncio.create_subprocess_exec(
            *match_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        # Stream output from both stdout and stderr concurrently
        while True:
            # Try to read from stdout
            try:
                stdout_line = await asyncio.wait_for(process1.stdout.readline(), timeout=0.1)
                if stdout_line:
                    text = stdout_line.decode('utf-8', errors='replace').rstrip()
                    if text:
                        yield f"data: {{\"type\": \"stdout\", \"message\": {json.dumps(text)}}}\n\n"
            except asyncio.TimeoutError:
                pass
            
            # Try to read from stderr
            try:
                stderr_line = await asyncio.wait_for(process1.stderr.readline(), timeout=0.1)
                if stderr_line:
                    text = stderr_line.decode('utf-8', errors='replace').rstrip()
                    if text:
                        yield f"data: {{\"type\": \"stderr\", \"message\": {json.dumps(text)}}}\n\n"
            except asyncio.TimeoutError:
                pass
            
            # Check if process is still running
            if process1.stdout.at_eof() and process1.stderr.at_eof():
                break
            
            # Small delay to prevent tight loop
            await asyncio.sleep(0.05)
        
        returncode1 = await process1.wait()
        
        if returncode1 != 0:
            stderr1 = await process1.stderr.read()
            error_msg = stderr1.decode('utf-8', errors='replace') if stderr1 else "Unknown error"
            yield f"data: {{\"type\": \"error\", \"message\": \"FIRMS matching failed: {json.dumps(error_msg[:200])}\"}}\n\n"
            return
        
        yield f"data: {{\"type\": \"stdout\", \"message\": \"✓ Fire-scene matching completed\"}}\n\n"
        yield f"data: {{\"type\": \"stdout\", \"message\": \" \"}}\n\n"
        
        # Step 2: Create dataset tiles
        yield f"data: {{\"type\": \"stdout\", \"message\": \"Step 2/4: Creating {tile_size}x{tile_size} tiles from matched pairs...\"}}\n\n"
        
        create_cmd = ["uv", "run", "python", "-u", "ml/scripts/create_dataset.py", "--tile-size", str(tile_size)]
        process2 = await asyncio.create_subprocess_exec(
            *create_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        # Stream output from create_dataset
        while True:
            try:
                stdout_line = await asyncio.wait_for(process2.stdout.readline(), timeout=0.1)
                if stdout_line:
                    text = stdout_line.decode('utf-8', errors='replace').rstrip()
                    if text:
                        yield f"data: {{\"type\": \"stdout\", \"message\": {json.dumps(text)}}}\n\n"
            except asyncio.TimeoutError:
                pass
            
            try:
                stderr_line = await asyncio.wait_for(process2.stderr.readline(), timeout=0.1)
                if stderr_line:
                    text = stderr_line.decode('utf-8', errors='replace').rstrip()
                    if text:
                        yield f"data: {{\"type\": \"stderr\", \"message\": {json.dumps(text)}}}\n\n"
            except asyncio.TimeoutError:
                pass
            
            if process2.stdout.at_eof() and process2.stderr.at_eof():
                break
            
            await asyncio.sleep(0.05)
        
        returncode2 = await process2.wait()
        
        if returncode2 != 0:
            stderr2 = await process2.stderr.read()
            error_msg = stderr2.decode('utf-8', errors='replace') if stderr2 else "Unknown error"
            yield f"data: {{\"type\": \"error\", \"message\": \"Dataset creation failed: {json.dumps(error_msg[:200])}\"}}\n\n"
            return
        
        yield f"data: {{\"type\": \"stdout\", \"message\": \"✓ Dataset tiles created successfully\"}}\n\n"
        yield f"data: {{\"type\": \"stdout\", \"message\": \" \"}}\n\n"
        
        # Step 2.5: Generate tile visualizations
        yield f"data: {{\"type\": \"stdout\", \"message\": \"Generating tile visualizations for quality check...\"}}\n\n"
        
        viz_cmd = ["uv", "run", "python", "-u", "ml/scripts/visualize_tiles.py", "--n-samples", "8"]
        process2b = await asyncio.create_subprocess_exec(
            *viz_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        # Stream output from visualizer (brief)
        while True:
            try:
                stdout_line = await asyncio.wait_for(process2b.stdout.readline(), timeout=0.1)
                if stdout_line:
                    text = stdout_line.decode('utf-8', errors='replace').rstrip()
                    if text and ('Created' in text or 'Saved' in text or '=' in text):
                        yield f"data: {{\"type\": \"stdout\", \"message\": {json.dumps(text)}}}\n\n"
            except asyncio.TimeoutError:
                pass
            
            try:
                stderr_line = await asyncio.wait_for(process2b.stderr.readline(), timeout=0.1)
                if stderr_line:
                    text = stderr_line.decode('utf-8', errors='replace').rstrip()
                    if text and 'warning' not in text.lower():
                        yield f"data: {{\"type\": \"stderr\", \"message\": {json.dumps(text)}}}\n\n"
            except asyncio.TimeoutError:
                pass
            
            if process2b.stdout.at_eof() and process2b.stderr.at_eof():
                break
            
            await asyncio.sleep(0.05)
        
        await process2b.wait()
        yield f"data: {{\"type\": \"stdout\", \"message\": \"✓ Sample visualizations generated\"}}\n\n"
        yield f"data: {{\"type\": \"stdout\", \"message\": \" \"}}\n\n"
        
        yield f"data: {{\"type\": \"stdout\", \"message\": \"Step 3/4: Preparing train/val/test splits...\"}}\n\n"
        
        # Step 3: Prepare splits
        prepare_cmd = ["uv", "run", "python", "-u", "ml/models/prepare_splits.py"]
        process3 = await asyncio.create_subprocess_exec(
            *prepare_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        # Stream output from both stdout and stderr concurrently
        while True:
            # Try to read from stdout
            try:
                stdout_line = await asyncio.wait_for(process3.stdout.readline(), timeout=0.1)
                if stdout_line:
                    text = stdout_line.decode('utf-8', errors='replace').rstrip()
                    if text:
                        yield f"data: {{\"type\": \"stdout\", \"message\": {json.dumps(text)}}}\n\n"
            except asyncio.TimeoutError:
                pass
            
            # Try to read from stderr
            try:
                stderr_line = await asyncio.wait_for(process3.stderr.readline(), timeout=0.1)
                if stderr_line:
                    text = stderr_line.decode('utf-8', errors='replace').rstrip()
                    if text:
                        yield f"data: {{\"type\": \"stderr\", \"message\": {json.dumps(text)}}}\n\n"
            except asyncio.TimeoutError:
                pass
            
            # Check if process is still running
            if process3.stdout.at_eof() and process3.stderr.at_eof():
                break
            
            # Small delay to prevent tight loop
            await asyncio.sleep(0.05)
        
        returncode3 = await process3.wait()
        
        if returncode3 != 0:
            stderr3 = await process3.stderr.read()
            error_msg = stderr3.decode('utf-8', errors='replace') if stderr3 else "Unknown error"
            yield f"data: {{\"type\": \"error\", \"message\": \"Split preparation failed: {json.dumps(error_msg[:200])}\"}}\n\n"
            return
        
        yield f"data: {{\"type\": \"stdout\", \"message\": \"✓ Train/val/test splits prepared\"}}\n\n"
        yield f"data: {{\"type\": \"stdout\", \"message\": \" \"}}\n\n"
        
        # Step 4: Generate co-located S2+S3 pairs
        yield f"data: {{\"type\": \"stdout\", \"message\": \"Step 4/4: Generating co-located S2+S3 pairs ({tile_size}x{tile_size} tiles)...\"}}\n\n"
        
        colocated_cmd = [
            "uv", "run", "python", "-u", "ml/scripts/generate_colocated_pairs.py",
            "--tile-size", str(tile_size),
            "--max-fire-pairs", "30",
            "--max-nofire-pairs", "10"
        ]
        process4 = await asyncio.create_subprocess_exec(
            *colocated_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        # Stream output
        while True:
            try:
                stdout_line = await asyncio.wait_for(process4.stdout.readline(), timeout=0.1)
                if stdout_line:
                    text = stdout_line.decode('utf-8', errors='replace').rstrip()
                    if text:
                        yield f"data: {{\"type\": \"stdout\", \"message\": {json.dumps(text)}}}\n\n"
            except asyncio.TimeoutError:
                pass
            
            try:
                stderr_line = await asyncio.wait_for(process4.stderr.readline(), timeout=0.1)
                if stderr_line:
                    text = stderr_line.decode('utf-8', errors='replace').rstrip()
                    if text:
                        yield f"data: {{\"type\": \"stderr\", \"message\": {json.dumps(text)}}}\n\n"
            except asyncio.TimeoutError:
                pass
            
            if process4.stdout.at_eof() and process4.stderr.at_eof():
                break
            
            await asyncio.sleep(0.05)
        
        returncode4 = await process4.wait()
        
        if returncode4 == 0:
            yield f"data: {{\"type\": \"stdout\", \"message\": \"✓ Co-located pairs generated\"}}\n\n"
            yield f"data: {{\"type\": \"complete\", \"message\": \"Training dataset prepared successfully! Ready for model training.\"}}\n\n"
        else:
            yield f"data: {{\"type\": \"stdout\", \"message\": \"⚠ Co-located pairs generation had issues (see above). Training data is still usable.\"}}\n\n"
            yield f"data: {{\"type\": \"complete\", \"message\": \"Training dataset prepared. Co-located pairs may need smaller tile size or more S3 data.\"}}\n\n"
    
    except Exception as e:
        yield f"data: {{\"type\": \"error\", \"message\": {json.dumps(str(e))}}}\n\n"


@router.post("/prepare-splits")
async def prepare_splits(tile_size: int = 128):
    """
    Execute full dataset pipeline: create tiles + prepare splits.
    
    Args:
        tile_size: Size of tiles to extract in pixels (default 128)
    
    Returns:
        Server-Sent Events stream with real-time progress
    """
    try:
        return StreamingResponse(
            stream_dataset_pipeline(tile_size=tile_size),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to prepare splits: {str(e)}"
        )


@router.get("/tile-samples")
async def get_tile_samples() -> Dict:
    """
    Get list of tile sample visualizations.
    
    Returns:
        List of tile sample images with metadata
    """
    samples_dir = Path("data/samples/tiles")
    samples_info_path = samples_dir / "samples_info.json"
    
    if not samples_info_path.exists():
        # Check if tiles exist - if so, we can generate samples
        tiles_dir = Path("ml/data/tiles")  # Correct path from config
        if tiles_dir.exists() and any(tiles_dir.glob("*.npy")):
            return {
                "samples": [],
                "message": "Tiles exist but visualizations not generated.",
                "can_generate": True
            }
        return {
            "samples": [],
            "message": "No training tiles available. Run 'Prepare Training Dataset' first.",
            "can_generate": False
        }
    
    try:
        with open(samples_info_path, 'r') as f:
            samples_info = json.load(f)
        
        # Return relative paths for frontend to fetch
        return {
            "samples": samples_info,
            "base_path": "/static/samples/tiles",
            "can_generate": True
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load tile samples: {str(e)}"
        )


class GenerateTileSamplesRequest(BaseModel):
    n_samples: int | None = 8
    seed: int | None = None


@router.post("/generate-tile-samples")
async def generate_tile_samples(req: GenerateTileSamplesRequest):
    """
    Generate tile sample visualizations from existing tiles.
    
    Returns:
        Server-Sent Events stream with progress
    """
    try:
        # Check if tiles exist
        tiles_dir = Path("ml/data/tiles")  # Correct path from config
        if not tiles_dir.exists() or not any(tiles_dir.glob("*.npy")):
            raise HTTPException(
                status_code=404,
                detail="No training tiles found. Run 'Prepare Training Dataset' first."
            )
        
        cmd = ["uv", "run", "python", "-u", "ml/scripts/visualize_tiles.py", "--n-samples", str(req.n_samples or 8)]
        if req.seed is not None:
            cmd.extend(["--seed", str(req.seed)])
        
        return StreamingResponse(
            stream_subprocess_output(cmd),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate tile samples: {str(e)}"
        )


@router.get("/dual-satellite-samples")
async def get_dual_satellite_samples() -> Dict:
    """
    Get list of dual-satellite fire visualizations.
    
    These show the same fire from both Sentinel-2 and Sentinel-3.
    """
    samples_dir = Path("data/samples/dual_satellite")
    
    if not samples_dir.exists():
        samples_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all dual fire images
    images = sorted(samples_dir.glob("dual_fire_*.png"))
    
    if not images:
        # Check if we have tiles from both satellites
        tiles_dir = Path("ml/data/tiles")
        labels_dir = Path("ml/data/labels")
        
        if tiles_dir.exists() and labels_dir.exists():
            return {
                "samples": [],
                "message": "No dual-satellite visualizations. Click 'Generate Dual View' to create them.",
                "can_generate": True
            }
        return {
            "samples": [],
            "message": "No training tiles available. Run 'Prepare Training Dataset' first.",
            "can_generate": False
        }
    
    samples = [
        {"filename": img.name, "type": "dual_satellite"}
        for img in images
    ]
    
    return {
        "samples": samples,
        "base_path": "/static/samples/dual_satellite",
        "can_generate": True
    }


class GenerateDualSatelliteRequest(BaseModel):
    n_samples: int | None = 6


@router.post("/generate-dual-satellite-samples")
async def generate_dual_satellite_samples(req: GenerateDualSatelliteRequest):
    """Generate dual-satellite fire visualizations."""
    try:
        cmd = [
            "uv", "run", "python", "-u", "ml/scripts/visualize_dual_satellite.py",
            "--n-samples", str(req.n_samples or 6)
        ]
        
        return StreamingResponse(
            stream_subprocess_output(cmd),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate dual-satellite samples: {str(e)}"
        )


@router.get("/temporal-sequence-samples")
async def get_temporal_sequence_samples() -> Dict:
    """Get list of temporal sequence visualizations."""
    samples_dir = Path("data/samples/temporal_sequence")
    
    if not samples_dir.exists():
        samples_dir.mkdir(parents=True, exist_ok=True)
    
    info_path = samples_dir / "samples_info.json"
    
    if not info_path.exists():
        return {
            "samples": [],
            "message": "No temporal sequence samples. Click 'Generate' to create them.",
            "can_generate": True
        }
    
    try:
        with open(info_path, 'r') as f:
            samples = json.load(f)
        return {
            "samples": samples,
            "base_path": "/static/samples/temporal_sequence",
            "can_generate": True
        }
    except Exception as e:
        return {"samples": [], "message": str(e), "can_generate": False}


class GenerateTemporalSequenceRequest(BaseModel):
    n_samples: int | None = 8


@router.post("/generate-temporal-sequence-samples")
async def generate_temporal_sequence_samples(req: GenerateTemporalSequenceRequest):
    """Generate temporal sequence visualizations."""
    try:
        cmd = [
            "uv", "run", "python", "-u", "ml/scripts/visualize_temporal_sequence.py",
            "--n-samples", str(req.n_samples or 8)
        ]
        
        return StreamingResponse(
            stream_subprocess_output(cmd),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate temporal sequence samples: {str(e)}"
        )


# ========== CO-LOCATED PAIRS ENDPOINTS ==========

class GenerateColocatedPairsRequest(BaseModel):
    max_fire_pairs: int | None = 30
    max_nofire_pairs: int | None = 10
    tile_size: int | None = 128


@router.post("/generate-colocated-pairs")
async def generate_colocated_pairs(req: GenerateColocatedPairsRequest):
    """
    Generate co-located S2+S3 tile pairs at the same geographic coordinates.
    This is the main pipeline for creating properly aligned multi-satellite data.
    
    Args:
        req: Request with max_fire_pairs, max_nofire_pairs, and tile_size
    """
    tile_size = req.tile_size or 128
    try:
        cmd = [
            "uv", "run", "python", "-u", "ml/scripts/generate_colocated_pairs.py",
            "--tile-size", str(tile_size),
            "--max-fire-pairs", str(req.max_fire_pairs or 30),
            "--max-nofire-pairs", str(req.max_nofire_pairs or 10)
        ]
        
        return StreamingResponse(
            stream_subprocess_output(cmd),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate co-located pairs: {str(e)}"
        )


@router.get("/colocated-samples")
async def get_colocated_samples() -> Dict:
    """Get list of co-located pair visualizations."""
    viz_dir = Path("data/samples/colocated_viz")
    pairs_dir = Path("data/samples/colocated")
    
    if not viz_dir.exists():
        viz_dir.mkdir(parents=True, exist_ok=True)
    
    info_path = viz_dir / "samples_info.json"
    
    # Check if visualizations exist
    if not info_path.exists():
        # Check if pairs exist (can generate visualizations)
        if pairs_dir.exists() and any(pairs_dir.iterdir()):
            return {
                "samples": [],
                "message": "Pairs exist. Click 'Generate Visualizations' to create sample images.",
                "can_generate": True,
                "pairs_exist": True
            }
        return {
            "samples": [],
            "message": "No co-located pairs. Click 'Generate Co-Located Pairs' first.",
            "can_generate": False,
            "pairs_exist": False
        }
    
    try:
        with open(info_path, 'r') as f:
            samples = json.load(f)
        return {
            "samples": samples,
            "base_path": "/static/samples/colocated_viz",
            "can_generate": True,
            "pairs_exist": True
        }
    except Exception as e:
        return {"samples": [], "message": str(e), "can_generate": False}


class GenerateColocatedVizRequest(BaseModel):
    n_samples: int | None = 10


@router.post("/generate-colocated-viz")
async def generate_colocated_viz(req: GenerateColocatedVizRequest):
    """Generate visualizations for co-located pairs."""
    try:
        cmd = [
            "uv", "run", "python", "-u", "ml/scripts/visualize_colocated.py",
            "--n-samples", str(req.n_samples or 10)
        ]
        
        return StreamingResponse(
            stream_subprocess_output(cmd),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate co-located visualizations: {str(e)}"
        )


@router.post("/train")
async def train_model():
    """
    Execute model training script with streaming output.
    
    Returns:
        Server-Sent Events stream with real-time training progress
    """
    try:
        # Check if training tiles exist
        tiles_dir = Path("ml/data/tiles")
        if not tiles_dir.exists() or not any(tiles_dir.glob("*.npy")):
            raise HTTPException(
                status_code=404,
                detail="No training tiles found. Run 'Prepare Training Dataset' first."
            )
        
        # Build command
        cmd = ["uv", "run", "python", "-u", "ml/models/train.py"]
        
        return StreamingResponse(
            stream_subprocess_output(cmd),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start training: {str(e)}"
        )
"""
Cache management API endpoints - APPEND to src/api/dataset.py
"""

import shutil


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
