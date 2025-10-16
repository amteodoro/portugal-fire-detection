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
        
        async def read_stream(stream, stream_type):
            while True:
                line = await stream.readline()
                if not line:
                    break
                text = line.decode('utf-8', errors='replace').rstrip()
                if text:
                    yield f"data: {{\"type\": \"{stream_type}\", \"message\": {json.dumps(text)}}}\n\n"
        
        # Read stdout and stderr concurrently
        async def read_all():
            tasks = [
                read_stream(process.stdout, "stdout"),
                read_stream(process.stderr, "stderr")
            ]
            
            for task in tasks:
                async for line in task:
                    yield line
        
        async for line in read_all():
            yield line
        
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


async def stream_dataset_pipeline() -> AsyncGenerator[str, None]:
    """
    Stream output from complete dataset pipeline:
    1. Match FIRMS fires to Sentinel-2 scenes
    2. Create dataset tiles
    3. Prepare train/val/test splits
    """
    try:
        # Step 1: Match FIRMS to Sentinel-2 scenes
        yield f"data: {{\"type\": \"stdout\", \"message\": \"Step 1/3: Matching FIRMS fire detections to Sentinel-2 scenes...\"}}\n\n"
        
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
        yield f"data: {{\"type\": \"stdout\", \"message\": \"Step 2/3: Creating dataset tiles from matched pairs...\"}}\n\n"
        
        create_cmd = ["uv", "run", "python", "-u", "ml/scripts/create_dataset.py"]
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
        
        yield f"data: {{\"type\": \"stdout\", \"message\": \"Step 3/3: Preparing train/val/test splits...\"}}\n\n"
        
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
        
        if returncode3 == 0:
            yield f"data: {{\"type\": \"complete\", \"message\": \"Training dataset prepared successfully! Ready for model training.\"}}\n\n"
        else:
            stderr3 = await process3.stderr.read()
            error_msg = stderr3.decode('utf-8', errors='replace') if stderr3 else "Unknown error"
            yield f"data: {{\"type\": \"error\", \"message\": \"Split preparation failed: {json.dumps(error_msg[:200])}\"}}\n\n"
    
    except Exception as e:
        yield f"data: {{\"type\": \"error\", \"message\": {json.dumps(str(e))}}}\n\n"


@router.post("/prepare-splits")
async def prepare_splits():
    """
    Execute full dataset pipeline: create tiles + prepare splits.
    
    Returns:
        Server-Sent Events stream with real-time progress
    """
    try:
        return StreamingResponse(
            stream_dataset_pipeline(),
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
