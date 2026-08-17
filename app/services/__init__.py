from app.services.base import BasePlateRecognizer
from app.services.factory import PlateRecognizerFactory
from app.services.strategies.nvidia_vision import NvidiaVisionStrategy
from app.services.strategies.plate_recognizer import PlateRecognizerStrategy
from app.services.strategies.tesseract_ocr import TesseractStrategy

__all__ = [
    "BasePlateRecognizer",
    "NvidiaVisionStrategy",
    "PlateRecognizerFactory",
    "PlateRecognizerStrategy",
    "TesseractStrategy",
]
