from pydantic import BaseModel
from typing import Optional

class FirePoint(BaseModel):
    id: str
    source: str  # "VIIRS" | "MODIS"
    latitude: float
    longitude: float
    acq_time: str  # ISO8601 UTC
    confidence: Optional[float] = None  # 0-100
    frp: Optional[float] = None  # Fire Radiative Power if available
