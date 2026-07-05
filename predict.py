# Spot the Fake Photo
# For each image it prints a score from 0 to 1.
# 0 = real photo, 1 = photo of a screen (recapture).

import os
from pathlib import Path

import numpy as np
from PIL import Image

# ----- Settings -----
IMAGE_DIR = "testing_dataset/screen_image_testing"   # folder of images to run on
MODEL_TYPE = "onnx"              # "onnx" (default) or "pt" (fallback, needs ultralytics + torch)
IMGSZ = 736                   # size the model actually uses (720 trained, rounded up to a multiple of 32)
CONFIDENCE_THRESHOLD = 0.5    # threshold for classifying an image as a screen image

# Class order the model was trained with: 0 = real_image, 1 = screen_image
CLASSES = {0: "real_image", 1: "screen_image"}

# model files live next to this script, inside artifacts/
ARTIFACTS = Path(__file__).parent / "artifacts"


def preprocess(image_path):
    # Same steps Ultralytics uses for classification:
    # resize the shorter side to IMGSZ, center crop to IMGSZ x IMGSZ, then scale to 0-1.
    img = Image.open(image_path).convert("RGB")

    w, h = img.size
    scale = IMGSZ / min(w, h)
    img = img.resize((round(w * scale), round(h * scale)), Image.BILINEAR)

    # center crop
    w, h = img.size
    left = (w - IMGSZ) // 2
    top = (h - IMGSZ) // 2
    img = img.crop((left, top, left + IMGSZ, top + IMGSZ))

    x = np.asarray(img, dtype=np.float32) / 255.0   # HWC, 0-1
    x = x.transpose(2, 0, 1)                         # to CHW
    x = np.expand_dims(x, 0)                         # add batch dim
    return x


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def run_onnx(image_paths):
    import onnxruntime as ort

    # load the model once
    session = ort.InferenceSession(str(ARTIFACTS / "model.onnx"))
    input_name = session.get_inputs()[0].name

    scores = {}
    for path in image_paths:
        x = preprocess(path)
        out = session.run(None, {input_name: x})[0][0]
        probs = softmax(out)
        scores[path] = float(probs[1])   # probability of screen_image
    return scores


def run_pt(image_paths):
    from ultralytics import YOLO

    # load the model once
    model = YOLO(str(ARTIFACTS / "model.pt"))

    scores = {}
    for path in image_paths:
        result = model.predict(path, imgsz=IMGSZ, verbose=False)[0]
        scores[path] = float(result.probs.data[1])   # probability of screen_image
    return scores


def main():
    # collect the images
    image_paths = []
    for name in sorted(os.listdir(IMAGE_DIR)):
        if name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
            image_paths.append(os.path.join(IMAGE_DIR, name))

    if MODEL_TYPE == "onnx":
        scores = run_onnx(image_paths)
    else:
        scores = run_pt(image_paths)

    # print a table: image name | class name | confidence
    print(f"{'Image Name':<40} {'Class Name':<20} {'Confidence':>10}")
    print("-" * 72)
    for path in image_paths:
        screen_prob = scores[path]                 # probability of screen_image
        is_screen = screen_prob >= CONFIDENCE_THRESHOLD
        class_name = "Photo of a Screen" if is_screen else "Real Photo"
        confidence = screen_prob if is_screen else 1.0 - screen_prob
        print(f"{os.path.basename(path):<40} {class_name:<20} {confidence:>10.4f}")


if __name__ == "__main__":
    main()
