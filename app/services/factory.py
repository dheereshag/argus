from typing import Dict, Type, List, Optional, Union
from app.core.config import settings
from app.core.logging import logger
from app.core.exceptions import ProviderNotFoundError
from app.schemas.plate import ProviderEnum
from app.services.base import BasePlateRecognizer
from app.services.strategies.plate_recognizer import PlateRecognizerStrategy
from app.services.strategies.nvidia_vision import NvidiaVisionStrategy
from app.services.strategies.paddle_ocr import PaddleOCRStrategy

class PlateRecognizerFactory:
    """
    Factory Pattern for selecting and instantiating ANPR Model Strategies dynamically.
    """

    _strategies: Dict[ProviderEnum, Type[BasePlateRecognizer]] = {
        ProviderEnum.PLATERECOGNIZER: PlateRecognizerStrategy,
        ProviderEnum.NVIDIA: NvidiaVisionStrategy,
        ProviderEnum.PADDLEOCR: PaddleOCRStrategy,
    }

    @classmethod
    def register_strategy(cls, provider: ProviderEnum, strategy_cls: Type[BasePlateRecognizer]):
        """
        Allows registering custom recognition strategies at runtime.
        """
        logger.info(f"Registering strategy for provider '{provider}': {strategy_cls.__name__}")
        cls._strategies[provider] = strategy_cls

    @classmethod
    def list_providers(cls) -> List[ProviderEnum]:
        return list(cls._strategies.keys())

    @classmethod
    def get_recognizer(cls, provider: Optional[Union[ProviderEnum, str]] = None) -> BasePlateRecognizer:
        """
        Factory method to return the appropriate strategy instance.
        If provider is omitted, uses DEFAULT_PROVIDER from settings.
        """
        target_provider = provider or settings.DEFAULT_PROVIDER

        if isinstance(target_provider, str):
            try:
                target_provider = ProviderEnum(target_provider.lower())
            except ValueError:
                logger.warning(f"Requested invalid provider name: '{target_provider}'")
                raise ProviderNotFoundError(
                    provider=target_provider,
                    available_providers=[p.value for p in cls.list_providers()]
                )

        if target_provider not in cls._strategies:
            logger.warning(f"Unregistered strategy requested: '{target_provider}'")
            raise ProviderNotFoundError(
                provider=target_provider.value,
                available_providers=[p.value for p in cls.list_providers()]
            )

        strategy_cls = cls._strategies[target_provider]
        logger.debug(f"Resolved strategy class '{strategy_cls.__name__}' for provider '{target_provider.value}'.")
        return strategy_cls()
