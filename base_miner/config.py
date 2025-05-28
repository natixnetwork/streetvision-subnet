from pathlib import Path
from typing import Any, Dict, List, Optional
import os

HUGGINGFACE_REPO = os.getenv("HUGGINGFACE_REPO", "no_huggingface_repo")
HUGGINGFACE_CACHE_DIR: Path = Path.home() / ".cache" / "huggingface"
IMAGE_DATASETS: Dict[str, List[Dict[str, str]]] = {
    "Roadwork": [
        {"path": f"{HUGGINGFACE_REPO}/roadwork"},
    ],
}