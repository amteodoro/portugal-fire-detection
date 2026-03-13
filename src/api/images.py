from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

router = APIRouter()

BASE_DIR = Path("/app/data").resolve()


@router.get("/{folder}/{filename}")
async def get_image(folder: str, filename: str):
    requested = (BASE_DIR / folder / filename).resolve()
    if not str(requested).startswith(str(BASE_DIR) + "/"):
        raise HTTPException(status_code=403, detail="Access denied")
    if not requested.exists():
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(requested)
