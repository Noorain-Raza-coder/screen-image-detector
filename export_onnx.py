# Export the trained .pt model to ONNX.
# Run this once to create artifacts/model.onnx

from ultralytics import YOLO

# Note: we trained at 720, but Ultralytics rounds the image size up to a
# multiple of 32, so the real size it uses is 736. We export at 736 so the
# ONNX model matches what the .pt model actually does.

PT_MODEL_PATH = "artifacts/model.pt"

model = YOLO(PT_MODEL_PATH)
model.export(format="onnx", imgsz=736)

# Ultralytics saves it as artifacts/model.onnx automatically
print("Done. Saved to artifacts/model.onnx")
