from pydantic import BaseModel
from typing import Optional

class ImageItem(BaseModel):
    id: str
    source: str  # "EUMETSAT_FRP" | "SENTINEL-2"
    datetime: str
    bbox: list[float]
    path: Optional[str] = None
    thumbnail_url: Optional[str] = None
    cloud_cover: Optional[float] = None
