"""
infer_yolov11.py
================
YOLOv11s Inference + ONNX Export
  - Runs person detection on image / video / webcam
  - Exports best.pt → best.onnx for pipeline deployment
  - Used as Stage 2 in the full surveillance pipeline
    (input: 640×640 frame from Zero-DCE++ or passthrough)
"""

import sys
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[2]
BEST_PT     = ROOT / 'models' / 'yolov11' / 'finetune_llvip_exdark' / 'weights' / 'best.pt'
ONNX_OUT    = ROOT / 'models' / 'yolov11' / 'best.onnx'


# ═══════════════════════════════════════════════════════════════════════════════
# Export to ONNX
# ═══════════════════════════════════════════════════════════════════════════════
def export_onnx(weights_path: Path = BEST_PT, output_path: Path = ONNX_OUT):
    from ultralytics import YOLO

    print(f"Exporting {weights_path} → ONNX...")
    model = YOLO(str(weights_path))
    model.export(
        format   = 'onnx',
        imgsz    = 640,
        opset    = 17,
        simplify = True,
        dynamic  = False,
    )
    # Ultralytics saves alongside .pt — move to our target path
    auto_out = weights_path.with_suffix('.onnx')
    if auto_out.exists() and auto_out != output_path:
        auto_out.rename(output_path)

    print(f"✅ ONNX saved → {output_path}")
    return output_path


# ═══════════════════════════════════════════════════════════════════════════════
# Inference Class (used by pipeline)
# ═══════════════════════════════════════════════════════════════════════════════
class YOLOv11Inference:
    """
    Pipeline-ready person detector.
    Input : 640×640 BGR frame (from Zero-DCE++ or passthrough)
    Output: list of dicts {bbox: [x1,y1,x2,y2], conf: float, class_id: int}
    """

    def __init__(
        self,
        weights_path: str = str(BEST_PT),
        conf_threshold: float = 0.35,
        iou_threshold:  float = 0.45,
        device: str = 'cuda',
    ):
        from ultralytics import YOLO
        self.model  = YOLO(weights_path)
        self.conf   = conf_threshold
        self.iou    = iou_threshold
        self.device = device
        print(f"[YOLOv11] Weights : {weights_path}")
        print(f"[YOLOv11] Conf    : {conf_threshold} | IoU: {iou_threshold}")

    def detect(self, bgr_frame: np.ndarray) -> list:
        """
        Args:
            bgr_frame: 640×640 BGR uint8 numpy array

        Returns:
            List of detections:
            [{'bbox': [x1,y1,x2,y2], 'conf': 0.87, 'class_id': 0, 'class_name': 'person'}, ...]
        """
        results = self.model.predict(
            source  = bgr_frame,
            conf    = self.conf,
            iou     = self.iou,
            device  = self.device,
            verbose = False,
        )

        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append({
                    'bbox'      : [int(x1), int(y1), int(x2), int(y2)],
                    'conf'      : float(box.conf[0]),
                    'class_id'  : int(box.cls[0]),
                    'class_name': r.names[int(box.cls[0])],
                })

        return detections

    def draw(self, bgr_frame: np.ndarray, detections: list) -> np.ndarray:
        """Draw bounding boxes on frame."""
        frame = bgr_frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf  = det['conf']
            label = f"{det['class_name']} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return frame


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default=str(BEST_PT))
    parser.add_argument('--input',   default=None, help='Image or video path')
    parser.add_argument('--output',  default='yolo_output')
    parser.add_argument('--conf',    type=float, default=0.35)
    parser.add_argument('--export',  action='store_true', help='Export to ONNX')
    args = parser.parse_args()

    # Export mode
    if args.export:
        export_onnx(Path(args.weights), ONNX_OUT)
        sys.exit(0)

    # Inference mode
    if not args.input:
        print("Provide --input <image_or_video> or use --export")
        sys.exit(1)

    detector   = YOLOv11Inference(args.weights, conf_threshold=args.conf)
    input_path = Path(args.input)

    # Image
    if input_path.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
        frame      = cv2.imread(str(input_path))
        detections = detector.detect(frame)
        output     = detector.draw(frame, detections)
        out_path   = f"{args.output}.jpg"
        cv2.imwrite(out_path, output)
        print(f"Detected {len(detections)} persons → {out_path}")
        for d in detections:
            print(f"  {d['class_name']} conf={d['conf']:.3f} bbox={d['bbox']}")

    # Video
    elif input_path.suffix.lower() in {'.mp4', '.avi', '.mov', '.mkv'}:
        cap    = cv2.VideoCapture(str(input_path))
        fps    = int(cap.get(cv2.CAP_PROP_FPS))
        w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        writer = cv2.VideoWriter(
            f"{args.output}.mp4",
            cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h)
        )
        for i in range(total):
            ret, frame = cap.read()
            if not ret:
                break
            dets   = detector.detect(frame)
            output = detector.draw(frame, dets)
            writer.write(output)
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{total} frames processed")
        cap.release()
        writer.release()
        print(f"Done → {args.output}.mp4")