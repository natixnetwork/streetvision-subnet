import os
import time
from typing import Dict, List, Optional
import requests
import bittensor as bt


class TaskTypesClient:
    
    def __init__(self, base_url: str = None, cache_duration: int = 300):
        self.base_url = base_url or os.environ.get("PROXY_CLIENT_URL", "https://hydra.natix.network")
        self.cache_duration = cache_duration
        self._cache: Optional[Dict] = None
        self._cache_timestamp: float = 0
        self._timeout = 10
    
    def _is_cache_valid(self) -> bool:
        return (
            self._cache is not None and 
            (time.time() - self._cache_timestamp) < self.cache_duration
        )
    
    def fetch_task_types(self, force_refresh: bool = False) -> Dict:
        if not force_refresh and self._is_cache_valid():
            bt.logging.debug("Using cached task types")
            return self._cache
        
        try:
            bt.logging.info(f"Fetching task types from {self.base_url}")
            response = requests.get(
                f"{self.base_url}/task_preferences/types",
                timeout=self._timeout,
                headers={"Accept": "application/json"}
            )
            response.raise_for_status()
            
            data = response.json()
            self._cache = data
            self._cache_timestamp = time.time()
            
            bt.logging.success(f"Successfully fetched task types from server")
            return data
            
        except requests.exceptions.RequestException as e:
            bt.logging.error(f"Failed to fetch task types from server: {e}")
            
            if self._cache is not None:
                bt.logging.warning("Using expired cache due to fetch failure")
                return self._cache
            
            return self._get_default_task_types()
        
        except Exception as e:
            bt.logging.error(f"Unexpected error fetching task types: {e}")
            return self._get_default_task_types()
    
    def _get_default_task_types(self) -> Dict:
        bt.logging.warning("Using default task types as fallback")
        
        return {
            "challenge_types": {
                1: "Roadwork"
            }
        }
    
    def get_challenge_type_mapping(self) -> Dict[int, str]:
        data = self.fetch_task_types()
        
        if "challenge_types" in data:
            return data["challenge_types"]
        
        return {1: "Roadwork"}
    
    def get_available_challenge_types(self) -> List[str]:
        data = self.fetch_task_types()
        
        if "challenge_types" in data:
            return list(data["challenge_types"].values())
        
        return ["Roadwork"]
    
    def refresh_cache(self) -> Dict:
        return self.fetch_task_types(force_refresh=True)


_global_client: Optional[TaskTypesClient] = None


def get_task_types_client(base_url: Optional[str] = None) -> TaskTypesClient:
    global _global_client
    
    if _global_client is None or (base_url and _global_client.base_url != base_url):
        _global_client = TaskTypesClient(base_url=base_url)
    
    return _global_client