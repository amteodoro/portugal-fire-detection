from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
import os

router = APIRouter()

@router.get("/{folder}/{filename}")
async def get_image(folder: str, filename: str):
    path = f"/app/data/{folder}/{filename}"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(path)
