from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from src.domain.enums import Capability
from src.domain.observation import Observation


class ProviderError(Exception):
    pass


class ProviderAuthError(ProviderError):
    pass


class ProviderUnreachableError(ProviderError):
    pass


class ProviderNotAvailableError(ProviderError):
    pass


class BaseSatelliteProvider(ABC):
    name: str
    capability: Capability

    @abstractmethod
    def search(
        self,
        bbox: tuple[float, float, float, float],
        date_range: tuple[str, str],
        limit: int = 10,
    ) -> list[Observation]:
        pass

    @abstractmethod
    def fetch_asset(self, observation: Observation, output_path: Path) -> Path:
        pass

    @abstractmethod
    def health_check(self) -> dict[str, Any]:
        pass
