"""Simple API key authentication used by the FastAPI app."""

from fastapi import Security, HTTPException
from fastapi.security.api_key import APIKeyHeader


API_KEY = "tym_razem_to_musi_poleciec"
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def get_api_key(api_key_header: str = Security(api_key_header)):
    """Validate provided API key or raise ``HTTPException``."""

    if api_key_header == API_KEY:
        return api_key_header
    raise HTTPException(status_code=403, detail="Invalid API Key")
