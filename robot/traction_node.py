#!/usr/bin/env python3
"""
traction_node.py

Subscribes to zed/image from zed_pub_node via commlink and runs
MobileNetV3-Small ONNX traction regression at ~30 Hz.

Requires:
    pip install onnxruntime (or onnxruntime-gpu)
    pip install joblib

Usage:
    python ~/robot/traction_node.py
    python ~/robot/traction_node.py --model ~/robot/model.onnx --scaler ~/robot/label_scaler.pkl
"""

import argparse
import time
import numpy as np
from commlink import Subscriber

# ── Config ───────────────────────────────────────────────────────────────────
ZED_PUB_HOST    = "localhost"
ZED_PUB_PORT    = 6000
IMAGE_TOPIC     = "zed/image"
IMG_SIZE        = 224

# ImageNet normalization (must match training exactly)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Traction buckets
def traction_category(t: float) -> str:
    if t < 1001:
        return "LOW  (slippery)"
    elif t < 1121:
        return "MEDIUM (normal)"
    else:
        return "HIGH (grippy)"


def preprocess(bgr: np.ndarray) -> np.ndarray:
    """Crop lower half, resize to 224x224, ImageNet-normalize → (1,3,224,224)."""
    import cv2
    h = bgr.shape[0]
    floor = bgr[h // 2:, :]                              # lower half = floor
    rgb   = cv2.cvtColor(floor, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    img = resized.astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    return img.transpose(2, 0, 1)[np.newaxis]            # (1, 3, H, W)


def main():
    parser = argparse.ArgumentParser(description="Traction inference node")
    parser.add_argument("--model",  default="model.onnx",       help="Path to ONNX model")
    parser.add_argument("--scaler", default="label_scaler.pkl",  help="Path to label scaler")
    parser.add_argument("--host",   default=ZED_PUB_HOST,        help="zed_pub_node host")
    parser.add_argument("--port",   default=ZED_PUB_PORT, type=int, help="zed_pub_node port")
    args = parser.parse_args()

    # ── Load model ───────────────────────────────────────────────────────────
    import onnxruntime as ort
    import joblib

    providers = ["CPUExecutionProvider"]
    session   = ort.InferenceSession(args.model, providers=providers)
    scaler    = joblib.load(args.scaler)

    active_provider = session.get_providers()[0]
    print(f"[Traction] ONNX running on: {active_provider}")
    print(f"[Traction] Subscribing to {IMAGE_TOPIC} @ {args.host}:{args.port}")

    # ── Subscribe to zed/image ────────────────────────────────────────────────
    sub = Subscriber(host=args.host, port=args.port, topics=[IMAGE_TOPIC])

    # Get input name from model
    input_name = session.get_inputs()[0].name    # usually "image"
    output_name = session.get_outputs()[0].name  # usually "traction"

    print("[Traction] Waiting for frames...")
    frame_count = 0
    t_start = time.time()

    try:
        while True:
            msg = sub.get(IMAGE_TOPIC)
            if msg is None:
                time.sleep(0.005)
                continue

            # msg = {"timestamp": ..., "image": np.ndarray (H,W,3) BGR}
            bgr = msg["image"]

            # Preprocess
            tensor = preprocess(bgr)

            # Infer
            pred_norm = session.run([output_name], {input_name: tensor})[0]

            # Inverse-transform to traction units
            traction = float(
                scaler.inverse_transform(pred_norm.reshape(-1, 1)).ravel()[0]
            )

            category = traction_category(traction)
            frame_count += 1
            elapsed = time.time() - t_start
            fps = frame_count / elapsed if elapsed > 0 else 0

            print(f"[Traction] {traction:7.1f}  |  {category:<22}  |  {fps:.1f} fps")

    except KeyboardInterrupt:
        print("[Traction] Shutting down.")
    finally:
        sub.stop()


if __name__ == "__main__":
    main()
