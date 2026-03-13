"""
Dataset training pipeline API endpoints.

Provides streaming endpoints for preparing training data splits, generating
tile/dual-satellite/temporal/colocated sample visualizations, and launching
model training.
"""

from fastapi import APIRouter, HTTPException
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


# ---------------------------------------------------------------------------
# Shared streaming helper
# ---------------------------------------------------------------------------

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
        queue: asyncio.Queue = asyncio.Queue()

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


# Common headers for SSE responses
COMMON_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
}


# ---------------------------------------------------------------------------
# Full pipeline helper (used by /prepare-splits)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Training pipeline endpoints
# ---------------------------------------------------------------------------

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
            headers=COMMON_HEADERS,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to prepare splits: {str(e)}"
        )


# ---------------------------------------------------------------------------
# Tile sample endpoints
# ---------------------------------------------------------------------------

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
        # Check if tiles exist in tiles_dir or in processed splits
        tiles_dir = Path("ml/data/tiles")
        splits_dir = Path("ml/data/processed/splits")
        has_tiles = (tiles_dir.exists() and any(tiles_dir.glob("*.npy"))) or (
            splits_dir.exists() and any(f for f in splits_dir.rglob("*.npy") if f.exists())
        )
        if has_tiles:
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
        # Check if tiles exist in tiles_dir or in processed splits
        tiles_dir = Path("ml/data/tiles")
        splits_dir = Path("ml/data/processed/splits")
        has_tiles = (tiles_dir.exists() and any(tiles_dir.glob("*.npy"))) or (
            splits_dir.exists() and any(f for f in splits_dir.rglob("*.npy") if f.exists())
        )
        if not has_tiles:
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
            headers=COMMON_HEADERS,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate tile samples: {str(e)}"
        )


# ---------------------------------------------------------------------------
# Dual-satellite sample endpoints
# ---------------------------------------------------------------------------

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
            headers=COMMON_HEADERS,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate dual-satellite samples: {str(e)}"
        )


# ---------------------------------------------------------------------------
# Temporal sequence sample endpoints
# ---------------------------------------------------------------------------

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
            headers=COMMON_HEADERS,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate temporal sequence samples: {str(e)}"
        )


# ---------------------------------------------------------------------------
# Co-located pairs endpoints
# ---------------------------------------------------------------------------

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
            headers=COMMON_HEADERS,
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
            headers=COMMON_HEADERS,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate co-located visualizations: {str(e)}"
        )


# ---------------------------------------------------------------------------
# Model training endpoint
# ---------------------------------------------------------------------------

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
            headers=COMMON_HEADERS,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start training: {str(e)}"
        )
