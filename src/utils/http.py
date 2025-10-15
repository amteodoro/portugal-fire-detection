import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

DEFAULT_TIMEOUT = httpx.Timeout(20.0, connect=10.0)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
async def fetch(url: str, *, params: dict | None = None, headers: dict | None = None) -> bytes:
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT, follow_redirects=True) as client:
        r = await client.get(url, params=params, headers=headers)
        r.raise_for_status()
        return r.content

async def fetch_json(url: str, *, params: dict | None = None, headers: dict | None = None):
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT, follow_redirects=True) as client:
        r = await client.get(url, params=params, headers=headers)
        r.raise_for_status()
        return r.json()
