"""
Stage 2: License Plate Text Recognition (OCR) and Spatial Candidate Selection.

Integrates RapidOCR (ONNX Runtime) with domain-specific post-processing:
  - 2D spatial centroid tracking for quad/box detections.
  - CLAHE (Contrast Limited Adaptive Histogram Equalization) and cubic upscaling for poor lighting.
  - Top-to-bottom vertical sorting and decal word filtering.
  - Two-line plate reconstruction via 2D Euclidean distance candidate pairing.
  - Two-pass recognition pipeline with automatic contrast enhancement fallback.
"""

import logging
import math
import re
from typing import Any

import cv2
import numpy as np
from PIL import Image
from rapidocr import RapidOCR

from app.constants import INDIAN_PLATE_REGEX
from app.core.contracts import require
from app.core.logging import logger
from app.schemas import OCRToken, PlateCandidate
from app.services.image_processing import ImageInput, load_rgb
from app.services.plate_rules import (
    is_decal_word,
    normalize_candidate_strings,
    parse_plate_info,
)

# Suppress verbose internal ONNX and font diagnostic logs from RapidOCR
_rapid_logger = logging.getLogger("RapidOCR")
_rapid_logger.setLevel(logging.ERROR)
for _h in _rapid_logger.handlers:
    _h.setLevel(logging.ERROR)


class PlateRecognizer:
    """
    ANPR Engine using RapidOCR ONNX Runtime with 2D spatial layout candidate pairing.

    Executes optical character recognition, parses candidate text tokens, reconstructs
    both single-line and stacked two-line Indian license plates, and validates them
    against national registration syntax.
    """

    _engine: RapidOCR | None = None

    @classmethod
    def get_engine(cls) -> RapidOCR:
        """
        Return the singleton RapidOCR engine instance, creating it on first access.

        Returns:
            RapidOCR: Initialized RapidOCR inference engine.
        """
        if cls._engine is None:
            cls._engine = RapidOCR()
        return cls._engine

    @classmethod
    def check_engine(cls) -> bool:
        """
        Verify that the RapidOCR engine is operational during service startup health checks.

        Returns:
            bool: True if the engine initializes successfully.
        """
        return cls.get_engine() is not None

    @staticmethod
    def _get_box_centroid(box: Any) -> tuple[float | None, float | None]:
        """
        Calculate the (x_center, y_center) centroid of an OCR bounding box.

        RapidOCR returns bounding boxes in varying representations:
          - 4-point quadrilateral polygon: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
          - 4-element bounding box: [x1, y1, x2, y2]

        Args:
            box: Bounding box or polygon coordinates returned by the OCR engine.

        Returns:
            tuple[float | None, float | None]: (x_centroid, y_centroid) or (None, None) if invalid.
        """
        if box is None:
            return None, None
        try:
            # Case 1: [x1, y1, x2, y2] box format
            if isinstance(box, (list, tuple, np.ndarray)) and len(box) >= 4:
                if all(isinstance(v, (int, float, np.number)) for v in box[:4]):
                    return float((box[0] + box[2]) / 2.0), float((box[1] + box[3]) / 2.0)
                # Case 2: [[x1, y1], [x2, y2], ...] 4-corner polygon format
                if all(isinstance(pt, (list, tuple, np.ndarray)) and len(pt) >= 2 for pt in box):
                    pts_x = [pt[0] for pt in box]
                    pts_y = [pt[1] for pt in box]
                    return float(np.mean(pts_x)), float(np.mean(pts_y))
        except (TypeError, ValueError, IndexError, AttributeError):
            pass
        return None, None

    @staticmethod
    def _enhance_contrast(img: Image.Image) -> Image.Image:
        """
        Enhance image contrast and resolution using CLAHE and bicubic upscaling.

        Applied during second-pass OCR fallback when a vehicle crop is low-contrast,
        shadowed, dirty, or distant:
          - Upscales small crops (< 600px) by 2.5x with bicubic interpolation.
          - Applies CLAHE (clipLimit=3.5, tileGridSize=(4, 4)) to boost plate embossed text.

        Args:
            img: Source PIL Image.

        Returns:
            Image.Image: Contrast-enhanced PIL RGB Image.
        """
        np_img = np.array(img)
        h, w = np_img.shape[:2]

        # Upscale low-resolution crops to provide sufficient pixel density for OCR text detector
        if w < 600 or h < 600:
            np_img = cv2.resize(np_img, (0, 0), fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY) if len(np_img.shape) == 3 else np_img
        clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(4, 4))
        enhanced_rgb = cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2RGB)
        return Image.fromarray(enhanced_rgb)

    def parse_plate_info(self, raw_plate: str | None) -> dict[str, Any] | None:
        """
        Validate candidate plate string against Indian plate regex and resolve State/UT.

        Args:
            raw_plate: Unprocessed candidate plate string.

        Returns:
            dict[str, Any] | None: Parsed plate information or None.
        """
        return parse_plate_info(raw_plate)

    def _run_ocr_inference(self, img_pil: Image.Image) -> list[OCRToken]:
        """Execute RapidOCR engine and return bounding tokens with centroids."""
        try:
            res = self.get_engine()(np.array(img_pil))
            txts = getattr(res, "txts", None) if res else None
            scores = getattr(res, "scores", None) if res else None
            if not txts or not scores:
                return []
            boxes = getattr(res, "boxes", None)
            tokens: list[OCRToken] = []
            for idx, (t, s) in enumerate(zip(txts, scores, strict=False)):
                cx, cy = self._get_box_centroid(boxes[idx]) if (boxes is not None and idx < len(boxes)) else (None, None)
                tokens.append(OCRToken(text=str(t), score=float(s), cx=cx, cy=cy))
            if any(item.cy is not None for item in tokens):
                tokens.sort(key=lambda x: x.cy if x.cy is not None else 9999.0)
            return tokens
        except (RuntimeError, ValueError, TypeError, IndexError, AttributeError, OSError) as e:
            logger.error(f"[ocr/rapidocr] OCR execution failed: {e}")
            return []

    @staticmethod
    def _clean_and_filter_tokens(raw_items: list[OCRToken]) -> tuple[list[OCRToken], str]:
        """Filter low-confidence tokens and decals, returning clean tokens and raw text summary."""
        clean_tokens: list[OCRToken] = []
        raw_text_parts: list[str] = []
        seen_tokens: set[str] = set()

        for token in raw_items:
            if not token.text or token.score < 0.20:
                continue
            raw_text_parts.append(token.text.strip())

            for chunk in [token.text, *token.text.split()]:
                cleaned = re.sub(r"[^A-Za-z0-9]", "", chunk).upper()
                if len(cleaned) >= 2 and not is_decal_word(cleaned) and cleaned not in seen_tokens:
                    seen_tokens.add(cleaned)
                    clean_tokens.append(OCRToken(text=cleaned, score=token.score, cx=token.cx, cy=token.cy))

        raw_summary = " ".join(raw_text_parts) if raw_text_parts else "N/A"
        return clean_tokens, raw_summary

    @staticmethod
    def _build_spatial_pairs(clean_tokens: list[OCRToken]) -> list[tuple[float, str, float]]:
        """Construct 2-line spatial candidate pairings sorted by Euclidean centroid distance."""
        candidate_pairs: list[tuple[float, str, float]] = []
        n = len(clean_tokens)
        for i in range(n):
            tok_a = clean_tokens[i]
            for j in range(i + 1, min(i + 6, n)):
                tok_b = clean_tokens[j]
                if tok_a.cx is not None and tok_a.cy is not None and tok_b.cx is not None and tok_b.cy is not None:
                    dist = math.hypot(tok_a.cx - tok_b.cx, tok_a.cy - tok_b.cy)
                    y_mean = float((tok_a.cy + tok_b.cy) / 2.0)
                else:
                    dist = float(abs(i - j) * 100.0)
                    y_mean = tok_a.cy or tok_b.cy or 0.0

                candidate_pairs.append((dist, tok_a.text + tok_b.text, y_mean))
                candidate_pairs.append((dist + 0.1, tok_b.text + tok_a.text, y_mean))

        candidate_pairs.sort(key=lambda p: p[0])
        return candidate_pairs

    @staticmethod
    def _collect_candidates(
        clean_tokens: list[OCRToken],
        candidate_pairs: list[tuple[float, str, float]],
        raw_text_summary: str,
    ) -> list[PlateCandidate]:
        """Evaluate single-line and paired candidates against Indian plate syntax."""
        plate_candidates: list[PlateCandidate] = []
        seen_matched_plates: set[str] = set()

        def _evaluate(raw_text: str, y_pos: float) -> None:
            for rank, cand_norm in enumerate(normalize_candidate_strings(raw_text)):
                match = INDIAN_PLATE_REGEX.fullmatch(cand_norm)
                if match:
                    info = parse_plate_info(match.group(0))
                    if info:
                        plate_num = info.get("plate")
                        if plate_num and plate_num not in seen_matched_plates:
                            seen_matched_plates.add(plate_num)
                            info["raw_text"] = raw_text_summary
                            plate_candidates.append(PlateCandidate(y_pos=y_pos, rank=rank, info=info))

        for tok in clean_tokens:
            _evaluate(tok.text, tok.cy or 0.0)

        for _, pair_raw, y_pos in candidate_pairs:
            _evaluate(pair_raw, y_pos)

        return plate_candidates

    def _extract_plates_from_image_array(self, img_pil: Image.Image) -> list[dict[str, Any]]:
        """
        Extract and validate Indian license plates from a PIL image array.

        Pipeline:
          1. RapidOCR inference to obtain text boxes, confidence scores, and raw strings.
          2. Vertical spatial sorting (top-to-bottom by centroid Y).
          3. Filtering low-confidence tokens (< 0.20) and commercial vehicle decal words.
          4. Single-line candidate evaluation with positional normalization.
          5. Two-line spatial reconstruction (Euclidean distance pairing for split plates).
          6. Priority ranking and deduplication.
        """
        require(img_pil is not None, "_extract_plates_from_image_array received None")

        raw_items = self._run_ocr_inference(img_pil)
        if not raw_items:
            return []

        clean_tokens, raw_summary = self._clean_and_filter_tokens(raw_items)
        pairs = self._build_spatial_pairs(clean_tokens)
        candidates = self._collect_candidates(clean_tokens, pairs, raw_summary)

        if candidates:
            candidates.sort(key=lambda c: (-c.rank, c.y_pos), reverse=True)
            return [candidates[0].info]

        return [{"plate": "N/A", "state": "N/A", "raw_text": raw_summary}]

    def recognize(
        self,
        image_input: ImageInput,
        filename: str = "image.jpg",
    ) -> list[dict[str, Any]]:
        """
        Process an image input with RapidOCR engine and automatic CLAHE enhancement fallback.

        Two-Pass Strategy:
          Pass 1: Direct OCR on the input RGB image.
          Pass 2: If Pass 1 detects no valid plate, enhance contrast and resolution via
                  CLAHE + bicubic upscaling and retry OCR.

        Args:
            image_input: Image representation (file path, raw bytes, PIL Image, or NumPy array).
            filename: Name of the file being processed for diagnostic logging.

        Returns:
            list[dict[str, Any]]: Extracted plate information dictionaries.
        """
        require(image_input is not None, "recognize() called with no image")
        pil_img = load_rgb(image_input)

        # Pass 1: Standard extraction on original image
        res = self._extract_plates_from_image_array(pil_img)
        if any(r.get("plate") and r.get("plate") != "N/A" for r in res):
            return res

        # Pass 2: Contrast and resolution enhancement fallback
        try:
            enhanced_img = self._enhance_contrast(pil_img)
            res_enh = self._extract_plates_from_image_array(enhanced_img)
            if any(r.get("plate") and r.get("plate") != "N/A" for r in res_enh):
                return res_enh
            if res_enh:
                return res_enh
        except (cv2.error, ValueError, RuntimeError, OSError, TypeError) as e:
            logger.debug(f"Contrast enhancement fallback skipped: {e}")

        return res
