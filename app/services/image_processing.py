import io
import os
from typing import Union, Optional, Tuple
import cv2
import numpy as np
from PIL import Image, ImageOps, ImageDraw

from app.core.config import settings
from app.core.exceptions import InvalidImageError, PayloadTooLargeError

# Decompression-bomb guard. Pillow's own default limit only emits a warning;
# this makes an oversized image raise instead. A ~200 KB crafted PNG can declare
# 40000x40000 and consume gigabytes on decode.
Image.MAX_IMAGE_PIXELS = settings.MAX_IMAGE_PIXELS


def decode_and_downscale(
    image_bytes: bytes,
    max_edge: Optional[int] = None,
) -> bytes:
    """
    Validate an uploaded image and return normalised JPEG bytes.

    Guards the decode against decompression bombs, applies EXIF orientation, and
    downscales so the longest edge is at most `max_edge`.

    Downscaling here is not only a memory measure. It also keeps every payload
    under the Plate Recognizer 3.5 MB ceiling, which the provider strategy would
    otherwise silently skip, making large images unrecognisable rather than slow.

    Raises PayloadTooLargeError if the declared pixel count exceeds the budget,
    InvalidImageError if the bytes are not a decodable image.
    """
    max_edge = max_edge or settings.MAX_IMAGE_EDGE_PX

    try:
        probe = Image.open(io.BytesIO(image_bytes))
        width, height = probe.size
    except Image.DecompressionBombError as exc:
        raise PayloadTooLargeError(f"Image dimensions exceed the permitted budget: {exc}")
    except Exception as exc:
        raise InvalidImageError(f"Could not decode uploaded image: {exc}")

    if width * height > settings.MAX_IMAGE_PIXELS:
        raise PayloadTooLargeError(
            f"Image is {width}x{height} ({width * height} pixels); "
            f"limit is {settings.MAX_IMAGE_PIXELS} pixels."
        )

    try:
        pil_img = ImageOps.exif_transpose(probe).convert("RGB")
    except Image.DecompressionBombError as exc:
        raise PayloadTooLargeError(f"Image dimensions exceed the permitted budget: {exc}")
    except Exception as exc:
        raise InvalidImageError(f"Could not decode uploaded image: {exc}")

    if max(pil_img.size) > max_edge:
        pil_img.thumbnail((max_edge, max_edge), Image.LANCZOS)

    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 contour points: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def warp_perspective_crop(img_bytes: bytes) -> bytes:
    """
    Detects perspective distortion / quad angle in an image crop and returns
    a perspective-warped, de-skewed frontal rectangular view as JPEG bytes.
    If no significant angle/distortion is detected, returns original JPEG bytes.
    """
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        return img_bytes

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Edge & Contour Detection for 4-point quadrilateral (plate or bumper bounds)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 50, 200)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # If a 4-point polygon quadrilateral is found with reasonable area (> 5% of crop area)
        if len(approx) == 4 and cv2.contourArea(approx) > (0.05 * w * h):
            pts = approx.reshape(4, 2)
            rect = order_points(pts)
            tl, tr, br, bl = rect

            width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
            width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
            max_width = max(int(width_a), int(width_b))

            height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
            height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
            max_height = max(int(height_a), int(height_b))

            if max_width > 20 and max_height > 10:
                dst = np.array([
                    [0, 0],
                    [max_width - 1, 0],
                    [max_width - 1, max_height - 1],
                    [0, max_height - 1]
                ], dtype="float32")

                M = cv2.getPerspectiveTransform(rect, dst)
                warped = cv2.warpPerspective(img, M, (max_width, max_height))

                success, encoded = cv2.imencode(".jpg", warped)
                if success:
                    return encoded.tobytes()

    # 2. Rotation De-skewing fallback using minAreaRect
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_c) > (0.05 * w * h):
            rect = cv2.minAreaRect(largest_c)
            angle = rect[-1]
            if angle < -45:
                angle = 90 + angle

            if abs(angle) > 3.0:
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

                success, encoded = cv2.imencode(".jpg", rotated)
                if success:
                    return encoded.tobytes()

    return img_bytes


def crop_image_roi(
    image_input: Union[str, bytes],
    vehicle_box: Optional[Tuple[int, int, int, int]] = None,
    bottom_crop_ratio: float = 0.50,
    bottom_roi_only: bool = True
) -> bytes:
    """
    Extract cropped image bytes from an image input.

    - If vehicle_box is provided: crops the vehicle bounding box first.
    - If bottom_roi_only is True: crops the bottom half (bottom_crop_ratio) of the vehicle crop (or full image).
    """
    if isinstance(image_input, bytes):
        pil_img = Image.open(io.BytesIO(image_input))
    else:
        with open(image_input, "rb") as f:
            pil_img = Image.open(io.BytesIO(f.read()))

    pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
    w, h = pil_img.size

    if vehicle_box and len(vehicle_box) == 4:
        x1, y1, x2, y2 = vehicle_box
        crop_box = (max(0, x1), max(0, y1), min(w, x2), min(h, y2))
        if crop_box[2] > crop_box[0] and crop_box[3] > crop_box[1]:
            cropped = pil_img.crop(crop_box)
            if bottom_roi_only:
                vw, vh = cropped.size
                v_crop_top = int(vh * (1.0 - bottom_crop_ratio))
                cropped = cropped.crop((0, v_crop_top, vw, vh))
            buf = io.BytesIO()
            cropped.save(buf, format="JPEG")
            return buf.getvalue()

    # Fallback if no vehicle_box: crop bottom half of full image
    crop_top = int(h * (1.0 - bottom_crop_ratio)) if bottom_roi_only else 0
    cropped = pil_img.crop((0, crop_top, w, h))
    buf = io.BytesIO()
    cropped.save(buf, format="JPEG")
    return buf.getvalue()


def save_debug_images(
    img_bytes: bytes,
    filename: str,
    vehicle_boxes: list,
    vehicle_type: str,
    output_dir: str = "eval_debug_crops",
    bottom_crop_ratio: float = 0.50
) -> None:
    """
    Save annotated YOLO box image, full vehicle crops, and bottom ROI crops for OCR debugging.
    """
    os.makedirs(output_dir, exist_ok=True)

    base_stem, _ = os.path.splitext(filename)
    pil_img = ImageOps.exif_transpose(Image.open(io.BytesIO(img_bytes))).convert("RGB")
    w, h = pil_img.size

    # 1. Annotated image showing YOLO bounding box(es)
    annotated = pil_img.copy()
    draw = ImageDraw.Draw(annotated)

    if vehicle_boxes:
        for idx, box in enumerate(vehicle_boxes):
            if len(box) == 4:
                x1, y1, x2, y2 = box
                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                label = f"{vehicle_type}" if len(vehicle_boxes) == 1 else f"{vehicle_type} #{idx+1}"
                draw.rectangle([x1, max(0, y1 - 20), x1 + len(label) * 9 + 6, max(0, y1)], fill="red")
                draw.text((x1 + 3, max(0, y1 - 18)), label, fill="white")
    else:
        draw.rectangle([10, 10, 150, 35], fill="gray")
        draw.text((15, 15), "No Vehicle Box", fill="white")

    annotated.save(os.path.join(output_dir, f"{base_stem}_yolo_boxes.jpg"))

    # 2. Save cropped vehicle image(s) and their bottom ROI crops (bumper level)
    if vehicle_boxes:
        for idx, box in enumerate(vehicle_boxes):
            if len(box) == 4:
                x1, y1, x2, y2 = box
                crop_box = (max(0, x1), max(0, y1), min(w, x2), min(h, y2))
                if crop_box[2] > crop_box[0] and crop_box[3] > crop_box[1]:
                    cropped_img = pil_img.crop(crop_box)
                    crop_suffix = "" if len(vehicle_boxes) == 1 else f"_{idx}"

                    # Full vehicle crop from YOLO
                    cropped_img.save(os.path.join(output_dir, f"{base_stem}_crop{crop_suffix}.jpg"))

                    # Bottom 1/3 ROI crop (bumper level)
                    vw, vh = cropped_img.size
                    v_top_1_3 = int(vh * (1.0 - (1.0 / 3.0)))
                    v_roi_1_3 = cropped_img.crop((0, v_top_1_3, vw, vh))
                    v_roi_1_3.save(os.path.join(output_dir, f"{base_stem}_crop_bottom_1_3_roi{crop_suffix}.jpg"))

                    # Bottom 1/2 ROI crop
                    v_top_1_2 = int(vh * (1.0 - 0.50))
                    v_roi_1_2 = cropped_img.crop((0, v_top_1_2, vw, vh))
                    v_roi_1_2.save(os.path.join(output_dir, f"{base_stem}_crop_bottom_1_2_roi{crop_suffix}.jpg"))

                    # Bottom 2/3 ROI crop
                    v_top_2_3 = int(vh * (1.0 - (2.0 / 3.0)))
                    v_roi_2_3 = cropped_img.crop((0, v_top_2_3, vw, vh))
                    v_roi_2_3.save(os.path.join(output_dir, f"{base_stem}_crop_bottom_2_3_roi{crop_suffix}.jpg"))
    else:
        # Fallback: if no vehicle box detected
        crop_top_1_3 = int(h * (1.0 - (1.0 / 3.0)))
        roi_1_3 = pil_img.crop((0, crop_top_1_3, w, h))
        roi_1_3.save(os.path.join(output_dir, f"{base_stem}_crop_bottom_1_3_roi.jpg"))

        crop_top_1_2 = int(h * (1.0 - 0.50))
        roi_1_2 = pil_img.crop((0, crop_top_1_2, w, h))
        roi_1_2.save(os.path.join(output_dir, f"{base_stem}_crop_bottom_1_2_roi.jpg"))

        crop_top_2_3 = int(h * (1.0 - (2.0 / 3.0)))
        roi_2_3 = pil_img.crop((0, crop_top_2_3, w, h))
        roi_2_3.save(os.path.join(output_dir, f"{base_stem}_crop_bottom_2_3_roi.jpg"))
