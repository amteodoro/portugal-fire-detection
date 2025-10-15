from pydantic import BaseModel

class FeatureCollection(BaseModel):
    type: str = "FeatureCollection"
    features: list[dict]
