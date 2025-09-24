from typing import Any, Dict, Optional
import requests, time
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from src.utils.config import settings

class SportsDataError(Exception):
    pass

class SportsDataClient:
    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None):
        self.base_url = (base_url or settings.base_url).rstrip("/")
        self.api_key = api_key or settings.api_key
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "Ocp-Apim-Subscription-Key": self.api_key
        })
        self.session.headers.update({"Accept-Encoding": "gzip, deflate"})

    def _url(self, endpoint: str) -> str:
        endpoint = endpoint.lstrip("/")
        return f"{self.base_url}/{endpoint}"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type((requests.RequestException, SportsDataError)),
        reraise=True
    )
    def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Any:
        url = self._url(endpoint)
        resp = self.session.get(url, params=params or {}, timeout=20)
        if resp.status_code == 429:
            retry_after = int(resp.headers.get("Retry-After", "1"))
            time.sleep(retry_after)
            raise SportsDataError(f"Rate limited: {retry_after}s sonra tekrar dene")
        if not resp.ok:
            raise SportsDataError(f"GET {url} {resp.status_code}: {resp.text[:200]}")
        return resp.json()
