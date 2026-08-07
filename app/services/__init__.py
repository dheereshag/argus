from app.services.base import BasePlateRecognizer
from app.services.factory import PlateRecognizerFactory
from app.services.strategies.nvidia_vision import NvidiaVisionStrategy
from app.services.strategies.paddle_ocr import PaddleOCRStrategy
from app.services.strategies.plate_recognizer import PlateRecognizerStrategy

__all__ = [
    "BasePlateRecognizer",
    "NvidiaVisionStrategy",
    "PaddleOCRStrategy",
    "PlateRecognizerFactory",
    "PlateRecognizerStrategy",
]
