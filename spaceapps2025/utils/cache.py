"""Simple file-based caching utilities."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class FileCache:
    """Persist objects to disk so expensive downloads can be reused."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _hash_key(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.root / digest

    def get_path(self, key: str, suffix: str) -> Path:
        """Return the path where a cached artifact should live."""
        path = self._hash_key(key).with_suffix(suffix)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def load_json(self, key: str) -> Any:
        path = self.get_path(key, ".json")
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def save_json(self, key: str, payload: Any) -> Path:
        path = self.get_path(key, ".json")
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def has(self, key: str, suffix: str) -> bool:
        return self.get_path(key, suffix).exists()

    def touch(self, key: str, suffix: str) -> Path:
        path = self.get_path(key, suffix)
        path.touch(exist_ok=True)
        return path
