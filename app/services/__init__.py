from app.services.factory import PlateRecognizerFactory
from app.services.base import BasePlateRecognizer
from app.services.strategies.plate_recognizer import PlateRecognizerStrategy
from app.services.strategies.nvidia_vision import NvidiaVisionStrategy
from app.services.strategies.paddle_ocr import PaddleOCRStrategy

__all__ = [
    "PlateRecognizerFactory",
    "BasePlateRecognizer",
    "PlateRecognizerStrategy",
    "NvidiaVisionStrategy",
    "PaddleOCRStrategy",
]
