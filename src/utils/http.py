import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

DEFAULT_TIMEOUT = httpx.Timeout(30.0, connect=10.0)
_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=DEFAULT_TIMEOUT, follow_redirects=True)
    return _client


async def close_client() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
async def fetch(url: str, *, params: dict | None = None,
                headers: dict | None = None) -> bytes:
    r = await _get_client().get(url, params=params, headers=headers)
    r.raise_for_status()
    return r.content


async def fetch_json(url: str, *, params: dict | None = None,
                     headers: dict | None = None):
    r = await _get_client().get(url, params=params, headers=headers)
    r.raise_for_status()
    return r.json()
