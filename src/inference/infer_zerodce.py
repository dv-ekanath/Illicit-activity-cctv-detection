"""
infer_zerodce.py
================
Zero-DCE++ Inference — Pipeline Stage 1
  • Uses Stage 2 weights (stage2_best.pth) at inference
  • Input frame processed at 256×256, output resized to 640×640 for YOLO
  • Only triggers if mean frame brightness < threshold
  • Drop-in class for the full surveillance pipeline
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
import torchvision.transforms as T
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[1]))
from training.zerodce_model import ZeroDCE


class ZeroDCEInference:
    """
    Stateless enhancer — load once, call enhance() per frame.

    Pipeline flow:
        raw BGR frame
            → brightness check (mean < threshold?)
            → [if dark] resize to 256×256 → enhance → resize to 640×640
            → [if bright] resize directly to 640×640 (passthrough)
            → return 640×640 BGR frame to YOLO
    """

    def __init__(
        self,
        weights_path: str,
        device: str = 'cuda',
        brightness_threshold: int = 80,
        output_size: int = 640,
        enhance_size: int = 256,
    ):
        self.device               = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.brightness_threshold = brightness_threshold
        self.output_size          = output_size
        self.enhance_size         = enhance_size

        self.model = ZeroDCE(num_iterations=8).to(self.device)
        ckpt = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(ckpt['state_dict'])
        self.model.eval()

        self.to_tensor = T.ToTensor()

        print(f"[ZeroDCE] Weights  : {weights_path}")
        print(f"[ZeroDCE] Device   : {self.device}")
        print(f"[ZeroDCE] Enhance  : {enhance_size}×{enhance_size} → {output_size}×{output_size}")
        print(f"[ZeroDCE] Threshold: brightness < {brightness_threshold}")

    # ── Public API ─────────────────────────────────────────────────────────────

    def enhance(self, bgr_frame: np.ndarray) -> tuple:
        """
        Args:
            bgr_frame: OpenCV BGR uint8 frame (any resolution)

        Returns:
            (output_frame, was_enhanced)
            output_frame : 640×640 BGR uint8 — ready for YOLO
            was_enhanced : True if Zero-DCE++ was applied
        """
        dark = self._is_dark(bgr_frame)

        if not dark:
            out = cv2.resize(bgr_frame, (self.output_size, self.output_size),
                             interpolation=cv2.INTER_LINEAR)
            return out, False

        # BGR → RGB tensor at 256×256
        rgb    = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        pil    = Image.fromarray(rgb).resize(
            (self.enhance_size, self.enhance_size), Image.BILINEAR
        )
        tensor = self.to_tensor(pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            with autocast('cuda'):
                enhanced, _ = self.model(tensor)   # (1, 3, 256, 256)

        # Resize enhanced 256 → 640
        enhanced_640 = F.interpolate(
            enhanced, size=(self.output_size, self.output_size),
            mode='bilinear', align_corners=False
        )

        # Tensor → BGR numpy
        out_np  = enhanced_640.squeeze(0).permute(1, 2, 0).cpu().numpy()
        out_np  = (out_np * 255).clip(0, 255).astype(np.uint8)
        out_bgr = cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)

        return out_bgr, True

    # ── Internal ───────────────────────────────────────────────────────────────

    def _is_dark(self, bgr_frame: np.ndarray) -> bool:
        gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        return float(gray.mean()) < self.brightness_threshold


# ═══════════════════════════════════════════════════════════════════════════════
# CLI — quick test on image or video
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights',    required=True)
    parser.add_argument('--input',      required=True)
    parser.add_argument('--output',     default='enhanced_out')
    parser.add_argument('--threshold',  type=int, default=80)
    args = parser.parse_args()

    enhancer   = ZeroDCEInference(args.weights, brightness_threshold=args.threshold)
    input_path = Path(args.input)

    # Image
    if input_path.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
        frame = cv2.imread(str(input_path))
        out, was_enhanced = enhancer.enhance(frame)
        out_path = f"{args.output}.jpg"
        cv2.imwrite(out_path, out)
        print(f"{'Enhanced' if was_enhanced else 'Passthrough'} → {out_path} "
              f"(size: {out.shape[1]}×{out.shape[0]})")

    # Video
    elif input_path.suffix.lower() in {'.mp4', '.avi', '.mov', '.mkv'}:
        cap    = cv2.VideoCapture(str(input_path))
        fps    = int(cap.get(cv2.CAP_PROP_FPS))
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        out_path = f"{args.output}.mp4"
        writer   = cv2.VideoWriter(
            out_path, cv2.VideoWriter_fourcc(*'mp4v'),
            fps, (enhancer.output_size, enhancer.output_size)
        )
        enhanced_count = 0
        for i in range(total):
            ret, frame = cap.read()
            if not ret:
                break
            out_frame, was_enhanced = enhancer.enhance(frame)
            if was_enhanced:
                enhanced_count += 1
            writer.write(out_frame)
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{total} frames | enhanced: {enhanced_count}")
        cap.release()
        writer.release()
        print(f"\nDone → {out_path}")
        print(f"Enhanced: {enhanced_count}/{total} ({100*enhanced_count/total:.1f}%)")