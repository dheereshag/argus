"""Unit and integration tests for debug crop and intermediate preprocessed image saving."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from PIL import Image

from app.core.config import settings
from app.schemas import DetectedVehicle
from app.services.debug import save_debug_crop
from app.services.ocr import PlateRecognizer
from app.services.pipeline.stages import _ocr_single_vehicle


def test_save_debug_crop_disabled_by_default(tmp_path: Path):
    settings.DEBUG_SAVE_CROPS = False
    settings.DEBUG_CROPS_DIR = str(tmp_path)
    img = Image.new("RGB", (50, 50), color="blue")
    result = save_debug_crop(img, "raw", "test.jpg", vehicle_idx=0)
    assert result is None
    assert len(list(tmp_path.iterdir())) == 0


def test_save_debug_crop_enabled(tmp_path: Path):
    settings.DEBUG_SAVE_CROPS = True
    settings.DEBUG_CROPS_DIR = str(tmp_path)
    img = Image.new("RGB", (60, 40), color="red")
    saved = save_debug_crop(img, "raw", "truck_front.jpg", vehicle_idx=1)
    assert saved is not None
    assert saved.exists()
    assert "truck_front_veh1_raw.jpg" in saved.name
    with Image.open(saved) as opened:
        assert opened.size == (60, 40)


def test_save_debug_crop_rgba_conversion(tmp_path: Path):
    settings.DEBUG_SAVE_CROPS = True
    settings.DEBUG_CROPS_DIR = str(tmp_path)
    img = Image.new("RGBA", (30, 30), color=(255, 0, 0, 128))
    saved = save_debug_crop(img, "enhanced", "car.png", vehicle_idx=0)
    assert saved is not None
    assert saved.exists()
    assert "car_veh0_enhanced.jpg" in saved.name
    with Image.open(saved) as opened:
        assert opened.mode == "RGB"


def test_save_debug_crop_exception_handling(monkeypatch):
    settings.DEBUG_SAVE_CROPS = True
    img = Image.new("RGB", (10, 10))
    monkeypatch.setattr(Path, "mkdir", MagicMock(side_effect=OSError("Disk full")))
    result = save_debug_crop(img, "raw", "fail.jpg")
    assert result is None


def test_pipeline_saves_raw_crop_when_enabled(tmp_path: Path):
    settings.DEBUG_SAVE_CROPS = True
    settings.DEBUG_CROPS_DIR = str(tmp_path)
    crop = Image.new("RGB", (100, 50), color="green")
    vehicle = DetectedVehicle("truck", (10, 10, 110, 60), crop, (10, 10, 110, 60))

    recognizer = MagicMock()
    recognizer.recognize.return_value = [{"plate": "MH12AB1234", "box": (0, 0, 50, 20)}]

    _ocr_single_vehicle(recognizer, vehicle, "weighbridge.jpg", idx=2)
    saved_files = list(tmp_path.glob("*_raw.jpg"))
    assert len(saved_files) == 1
    assert "weighbridge_veh2_raw.jpg" in saved_files[0].name


@patch("app.services.ocr.PlateRecognizer.get_engine")
def test_retry_contrast_saves_enhanced_crop_when_enabled(mock_get_engine, tmp_path: Path):
    settings.DEBUG_SAVE_CROPS = True
    settings.DEBUG_CROPS_DIR = str(tmp_path)

    # First pass yields no text, second pass (contrast retry) yields valid plate
    mock_eng = MagicMock()
    mock_eng.side_effect = [
        SimpleNamespace(txts=[], scores=[], boxes=[]),
        SimpleNamespace(txts=["DL01C1234"], scores=[0.95], boxes=[[[0, 0], [100, 0], [100, 30], [0, 30]]]),
    ]
    mock_get_engine.return_value = mock_eng

    recognizer = PlateRecognizer()
    crop = Image.new("RGB", (80, 40), color="gray")
    results = recognizer.recognize(crop, filename="night_shot.jpg", vehicle_idx=3)

    assert any(r.get("plate") == "DL01C1234" for r in results)
    enhanced_files = list(tmp_path.glob("*_enhanced.jpg"))
    assert len(enhanced_files) == 1
    assert "night_shot_veh3_enhanced.jpg" in enhanced_files[0].name
