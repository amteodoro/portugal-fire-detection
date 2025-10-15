import os
import time
from PIL import Image
from io import BytesIO

DATA_DIR = "/app/data"

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def timestamp_name(prefix: str, ext: str = "png") -> str:
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return f"{prefix}_{ts}.{ext}"

def save_image_bytes(img_bytes: bytes, subdir: str, filename: str | None = None, optimize: bool = True) -> str:
    base = os.path.join(DATA_DIR, subdir)
    ensure_dir(base)
    name = filename or timestamp_name(subdir)
    path = os.path.join(base, name)
    if optimize:
        try:
            im = Image.open(BytesIO(img_bytes))
            if im.mode in ("RGBA", "P"):
                im = im.convert("RGB")
            im.save(path, optimize=True, quality=85)
        except Exception:
            # If image optimization fails, save raw bytes
            with open(path, "wb") as f:
                f.write(img_bytes)
    else:
        with open(path, "wb") as f:
            f.write(img_bytes)
    return path
