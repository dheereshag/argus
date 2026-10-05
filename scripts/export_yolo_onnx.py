"""Export YOLO26 PyTorch weights to ONNX format for ONNX Runtime acceleration."""

import os

from ultralytics import YOLO


def export_model(
    pt_path: str = "yolo26n.pt",
    output_onnx: str = "yolo26n.onnx",
) -> str:
    """Load YOLO26 model and export to ONNX format."""
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"Source model '{pt_path}' not found.")
    model = YOLO(pt_path)
    exported_path = model.export(format="onnx")
    return str(exported_path or output_onnx)


if __name__ == "__main__":
    export_model()
