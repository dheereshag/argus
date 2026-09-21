"""Stage 2: License Plate Text Recognition (OCR) and Spatial Candidate Selection."""

from typing import Any

from PIL import Image

from app.core.contracts import require
from app.schemas import OCRToken, PlateCandidate
from app.services.image_processing import ImageInput, load_rgb
from app.services.ocr.candidates import collect_candidates
from app.services.ocr.engine import check_engine, get_engine, run_ocr_inference
from app.services.ocr.enhancer import enhance_contrast, retry_contrast
from app.services.ocr.extractor import extract_plates
from app.services.ocr.geometry import get_box_centroid, get_box_geometry
from app.services.ocr.pairing import build_spatial_pairs
from app.services.ocr.spatial import (
    cluster_horizontal_lines,
    compute_token_bounds,
    is_same_horizontal_line,
)
from app.services.ocr.tokens import clean_and_filter_tokens


class PlateRecognizer:
    """ANPR Engine using RapidOCR ONNX Runtime with 2D spatial layout candidate pairing."""

    get_engine = staticmethod(get_engine)
    check_engine = staticmethod(check_engine)
    _enhance_contrast = staticmethod(enhance_contrast)
    _get_box_geometry = staticmethod(get_box_geometry)
    _get_box_centroid = staticmethod(get_box_centroid)
    _is_same_horizontal_line = staticmethod(is_same_horizontal_line)
    _cluster_horizontal_lines = staticmethod(cluster_horizontal_lines)
    _build_spatial_pairs = staticmethod(build_spatial_pairs)
    _compute_token_bounds = staticmethod(compute_token_bounds)
    _clean_and_filter_tokens = staticmethod(clean_and_filter_tokens)

    @staticmethod
    def parse_plate_info(raw_plate: str | None) -> dict[str, Any] | None:
        import app.services.ocr as ocr_mod

        return ocr_mod.parse_plate_info(raw_plate)

    def _run_ocr_inference(self, img: Image.Image) -> list[OCRToken]:
        return run_ocr_inference(img, self.get_engine)

    def _collect_candidates(self, toks: list[OCRToken], l: list[list[OCRToken]], p: list[Any], s: str) -> list[PlateCandidate]:
        return collect_candidates(toks, l, p, s, self.parse_plate_info)

    def _extract_plates_from_image_array(self, img: Image.Image) -> list[dict[str, Any]]:
        return extract_plates(img, self.get_engine, self.parse_plate_info)

    def recognize(self, image_input: ImageInput, filename: str = "image.jpg") -> list[dict[str, Any]]:
        require(image_input is not None, "recognize() called with no image")
        img = load_rgb(image_input)
        res = self._extract_plates_from_image_array(img)
        if any(r.get("plate") and r.get("plate") != "N/A" for r in res):
            return res
        return retry_contrast(self, img) or res
