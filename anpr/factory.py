from typing import Dict, Type
from decouple import config
from anpr.base import BasePlateRecognizer
from anpr.strategies import PlateRecognizerStrategy, NvidiaVisionStrategy


class PlateRecognizerFactory:
    """
    Factory Pattern for selecting and instantiating ANPR Model Strategies dynamically.
    """

    _strategies: Dict[str, Type[BasePlateRecognizer]] = {
        "platerecognizer": PlateRecognizerStrategy,
        "nvidia": NvidiaVisionStrategy,
    }

    @classmethod
    def register_strategy(cls, name: str, strategy_cls: Type[BasePlateRecognizer]):
        """
        Allows registering custom or third-party recognition strategies at runtime.
        """
        cls._strategies[name.lower()] = strategy_cls

    @classmethod
    def get_recognizer(cls, provider: str = None) -> BasePlateRecognizer:
        """
        Factory method to return the appropriate strategy instance.
        If provider is not specified, uses DEFAULT_PROVIDER from .env environment variable.
        """
        if not provider:
            provider = config("DEFAULT_PROVIDER", default="platerecognizer")

        provider_key = provider.lower()
        if provider_key not in cls._strategies:
            available = ", ".join(cls._strategies.keys())
            raise ValueError(f"Unknown provider '{provider}'. Available providers: {available}")

        return cls._strategies[provider_key]()
