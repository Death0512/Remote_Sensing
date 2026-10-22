import os
from src.ingestion.providers.base import BaseSatelliteProvider, ProviderNotAvailableError
from src.ingestion.providers.cdse_provider import CdseProvider
from src.ingestion.providers.mock_provider import MockSatelliteProvider
from src.ingestion.providers.planetary_provider import PlanetaryComputerProvider


def get_provider(name: str, **kwargs) -> BaseSatelliteProvider:
    normalized = name.strip().lower()
    if normalized == "mock":
        env = kwargs.get("app_env") or os.getenv("APP_ENV", "development")
        if env not in ("test", "simulation"):
            raise RuntimeError("MockSatelliteProvider forbidden in production/development environment")
        return MockSatelliteProvider(**kwargs)
    elif normalized in ("planetary_computer", "planetary", "pc"):
        return PlanetaryComputerProvider(**kwargs)
    elif normalized in ("cdse", "copernicus"):
        return CdseProvider(**kwargs)
    else:
        raise ProviderNotAvailableError(f"Unknown satellite provider: {name}")


class ProviderFactory:
    @staticmethod
    def get_provider(name: str, **kwargs) -> BaseSatelliteProvider:
        return get_provider(name, **kwargs)
