from app.services.factory import PlateRecognizerFactory
from app.services.base import BasePlateRecognizer
from app.services.strategies.plate_recognizer import PlateRecognizerStrategy
from app.services.strategies.nvidia_vision import NvidiaVisionStrategy

__all__ = [
    "PlateRecognizerFactory",
    "BasePlateRecognizer",
    "PlateRecognizerStrategy",
    "NvidiaVisionStrategy",
]
