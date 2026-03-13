"""
Dataset scene download API endpoints.

Provides streaming endpoints for downloading Sentinel-2, Sentinel-3,
FIRMS historical data, and OpenEO data.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pathlib import Path
import json
import sys
import asyncio
import re
from typing import Dict, List, Optional, AsyncGenerator

router = APIRouter()

# Add ml directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

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
# Fetch/download endpoints
# ---------------------------------------------------------------------------

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
            headers=COMMON_HEADERS,
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
            for p in request.products:
                if not re.match(r'^[A-Za-z0-9_\-]+$', p):
                    raise HTTPException(status_code=400, detail=f"Invalid product name: {p}")
            cmd.extend(["--products"] + request.products)

        if request.region:
            cmd.extend(["--region", request.region])

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
            headers=COMMON_HEADERS,
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
            headers=COMMON_HEADERS,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to execute Sentinel-3 download: {str(e)}"
        )
