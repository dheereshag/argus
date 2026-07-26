import io
import os
from typing import Union, Optional, Tuple
from PIL import Image, ImageOps, ImageDraw


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
