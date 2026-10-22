import hashlib
import time
from pathlib import Path
from typing import Any, Dict, Optional
import requests
import yaml


class WeightDownloader:
    def __init__(
        self,
        registry_path: str = "configs/model_registry.yaml",
        weights_dir: Optional[str] = None,
    ) -> None:
        self.registry_path = Path(registry_path)
        self.weights_dir = Path(weights_dir) if weights_dir else None
        self.registry: Dict[str, Any] = {}
        if self.registry_path.exists():
            with open(self.registry_path, "r", encoding="utf-8") as f:
                self.registry = yaml.safe_load(f) or {}

    def ensure_weight(
        self,
        model_name: str,
        max_retries: int = 3,
        backoff_factor: float = 0.5,
    ) -> Path:
        models = self.registry.get("models", {})
        if model_name not in models:
            raise KeyError(f"Model '{model_name}' not found in registry {self.registry_path}")

        meta = models[model_name]
        configured_path = Path(meta["local_path"])
        if self.weights_dir:
            target_path = self.weights_dir / configured_path.name
        else:
            target_path = configured_path

        if target_path.exists():
            return target_path

        target_path.parent.mkdir(parents=True, exist_ok=True)
        url = meta["url"]
        expected_sha256 = meta.get("sha256")
        part_path = target_path.with_suffix(target_path.suffix + ".part")

        for attempt in range(1, max_retries + 1):
            try:
                if part_path.exists():
                    part_path.unlink()

                hasher = hashlib.sha256() if expected_sha256 else None
                with requests.get(url, stream=True, timeout=30.0) as resp:
                    resp.raise_for_status()
                    with open(part_path, "wb") as f:
                        for chunk in resp.iter_content(chunk_size=65536):
                            if chunk:
                                f.write(chunk)
                                if hasher:
                                    hasher.update(chunk)

                if hasher and expected_sha256:
                    actual_sha256 = hasher.hexdigest().lower()
                    if actual_sha256 != expected_sha256.lower():
                        if part_path.exists():
                            part_path.unlink()
                        raise ValueError(
                            f"SHA256 mismatch for {model_name}: expected {expected_sha256}, got {actual_sha256}"
                        )

                part_path.replace(target_path)
                return target_path
            except Exception as exc:
                if part_path.exists():
                    part_path.unlink()
                if attempt == max_retries:
                    raise exc
                time.sleep(backoff_factor * (2 ** (attempt - 1)))

        return target_path
