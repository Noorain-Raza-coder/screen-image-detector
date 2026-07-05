# Screen Image Detector

Detect whether an uploaded product photo is a **real photo** or a **photo of a screen**
(a recaptured / "screen-shot-of-a-screen" image).

---

## 1. The Problem

In field-data collection apps, users are asked to **upload a real photo** of a product on a
shelf. Some users commit fraud by **photographing an existing image displayed on a screen**
(laptop, monitor, phone) and submitting that screen-recapture instead of taking a genuine
photo on-site.

Visually these two are very close, so we need an automatic classifier that flags recaptured
(screen) images before they are accepted into the pipeline.

**Goal:** given an image, output whether it is a `Real Photo` or a `Photo of a Screen`,
with a confidence score.

---

## 2. The Solution

A lightweight **image classification model** (Ultralytics YOLO classification head) trained on
a curated dataset of real product photos vs. photos of those products displayed on screens.

- **Two output classes:** `real_image` (0) and `screen_image` (1).
- **Two runtime backends:**
  - **ONNX** (default) — no PyTorch needed, fast startup, ideal for on-device / CPU.
  - **PyTorch (.pt)** — the native Ultralytics model, used for training, export, and GPU inference.
- Final model: **YOLO26s (classification)** — best accuracy among the variants tested.
- **Accuracy: ~93–95%** (47/49 correct on the held-out test set; 2 misclassified).

---

## 3. End-to-End Pipeline (what was actually done)

### 3.1 Data Collection
- Collected **daily consumer / FMCG product images** from different **kirana stores and marts**.
- **244 real product images**, captured in varied **lighting conditions, angles, and backgrounds**.
- To build the *screen* class, these images were **displayed on multiple different systems/screens**
  and **re-photographed**, introducing variety in screen type, moiré, glare, and reflections.
- Result: **~244 real + ~246 screen images** (see `dataset/real_image/` and `dataset/screen_image/`).

### 3.2 Dataset Preparation (Roboflow)
- Uploaded images to **Roboflow**.
- Performed **train / validation split**.
- Applied **augmentations** to improve robustness:
  - Blur
  - Brightness
  - Rotation
  - Saturation
  - Shear
- Exported the dataset with a **3× multiplier** (augmented copies), tripling the effective
  training set size.
- Final training layout (Ultralytics classification format):
  ```
  train/
    real_image/
    screen_image/
  val/
    real_image/
    screen_image/
  ```

### 3.3 Model Training
- Trained and compared multiple architectures:
  - YOLO11 nano, YOLO11 small
  - YOLO26 nano, **YOLO26 small ✅ (winner)**
- Training config (`training.py`): `epochs=250`, `batch=16`, `imgsz=720`, `lr0=0.001`,
  `patience=25`, `device=0` (GPU).
- **YOLO26s** performed best on evaluation and was selected as the production model.

> Note: Ultralytics rounds `imgsz` up to the nearest multiple of 32, so although training was
> requested at **720**, the model actually runs at **736**. Inference and ONNX export both use **736**.

### 3.4 Testing / Evaluation
- Evaluated on **49 held-out real + screen images** (`testing_dataset/screen_image_testing/`).
- **2 wrong predictions → ~93–95% accuracy.**

### 3.5 Export to ONNX
- The best `.pt` model is exported to ONNX (`export_onnx.py`) at `imgsz=736` for lightweight,
  torch-free deployment → `artifacts/model.onnx`.

---

## 4. Where Models & Data Are Stored

| Path | Contents |
|------|----------|
| `artifacts/model.pt` | Production PyTorch model (YOLO26s-cls) used by the `pt` backend |
| `artifacts/model.onnx` | Exported ONNX model used by the default `onnx` backend |
| `runs/classify/*/weights/best.pt` | Best checkpoints from each training run (e.g. `yolo26s-720-weights/weights/best.pt`) |
| `runs/classify/prediction_results/` | Saved prediction outputs per model |
| `dataset/real_image/`, `dataset/screen_image/` | Raw collected images (source dataset) |
| `roboflow_training_dataset/train`, `.../val` | Roboflow-split + augmented training data |
| `testing_dataset/screen_image_testing/` | Held-out test images |
| `Dataset_structure/` | Organized raw captures grouped by product & screen source |
---

## 5. Installation

Requires **Python 3.10** (developed in a conda environment).

```bash
# create & activate environment
conda create -n env-yolo python=3.10
conda activate env-yolo

# install dependencies
pip install -r requirements.txt
```

`requirements.txt` layers:
- **Inference only (ONNX, lightweight):** `onnxruntime`, `numpy`, `Pillow`
- **Training / export / PT backend:** `ultralytics`, `torch`, `torchvision`, `onnx`, `opencv-python`

If you only run ONNX inference, you can skip the heavy training group.

---

## 6. Running the Pipeline

### 6.1 Inference — `predict.py`

`predict.py` scores each image and prints a table:

```
Image Name                               Class Name           Confidence
------------------------------------------------------------------------
product_001.jpg                          Photo of a Screen        0.9873
product_002.jpg                          Real Photo               0.9541
```

- **Class name:** `Real Photo` if `screen_probability < 0.5`, else `Photo of a Screen`.
- **Confidence:** probability of the predicted class.

**Configure the run** by editing the settings at the top of `predict.py`:

```python
IMAGE_DIR = "testing_dataset/screen_image_testing"  # folder of images to score
MODEL_TYPE = "onnx"                                  # "onnx" (default) or "pt"
IMGSZ = 736                                           # inference image size (must match export)
CONFIDENCE_THRESHOLD = 0.5                            # screen vs. real decision boundary
```

Then run:

```bash
python predict.py
```

**Input:** `predict.py` scans a **directory** (`IMAGE_DIR`) and processes every image in it
(`.jpg .jpeg .png .bmp .webp`).
To score a **single image**, point `IMAGE_DIR` at a folder containing just that one image.

**Choosing the model type (ONNX vs PT):**
- `MODEL_TYPE = "onnx"` → uses `artifacts/model.onnx` (default, no torch required).
- `MODEL_TYPE = "pt"` → uses `artifacts/model.pt` (needs `ultralytics` + `torch`; enables GPU).

### 6.2 Training — `training.py`
```bash
python training.py
```
Trains YOLO26s-cls on the dataset in `roboflow_downloaded_dataset` (update the
`TRAINING_DATASET_PATH` constant to match your local dataset folder, e.g.
`roboflow_training_dataset`). Best weights land in `runs/classify/.../weights/best.pt`.

### 6.3 Export to ONNX — `export_onnx.py`
```bash
python export_onnx.py
```
Exports `artifacts/model.pt` → `artifacts/model.onnx` at `imgsz=736`.


## 7. Input / Output Summary

| | |
|---|---|
| **Input** | Product image(s): `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp` |
| **Preprocessing** | Resize shorter side to 736 → center-crop 736×736 → scale to 0–1 |
| **Output** | Per image: **image name**, **class** (`Real Photo` / `Photo of a Screen`), **confidence** |
| **Decision rule** | `screen_probability ≥ 0.5` → `Photo of a Screen` |

---

## 8. Performance

### 8.1 Latency

Two backends are supported (`predict.py` defaults to ONNX):

| Backend | Device | Avg latency / image | Throughput | Model load time |
|---------|--------|---------------------|------------|-----------------|
| ONNX (default) | CPU | 154.74 ms (p50 158.57, p95 179.47) | 6.46 img/sec | 64.9 ms |
| PyTorch (.pt) | CPU | ~132.1 ms (51.4 ms preprocess + 80.7 ms inference) | ~7.57 img/sec | 6,500.2 ms |
| PyTorch (.pt) | GPU (NVIDIA L4) | ~99.9 ms (min 64.2, p95 119.6) | 10.01 img/sec | — |

**GPU memory footprint:** ~200 MB during inference (yolo26s-cls, L4). This is small relative to
the L4's 24 GB — a single GPU could run several concurrent model instances or larger batch sizes.

### 8.2 Cost per Image

**On-device (phone, CPU):** effectively **$0 per image** — inference runs locally via the ONNX
model, with no server round-trip or per-request billing.

**Cloud server (GPU, NVIDIA L4), if run centrally instead of on-device:**
Using measured throughput of **10.01 images/sec** and an on-demand rate of ~**$0.44–$0.80/GPU-hour**:

```
Images/hour   = 10.01 × 3600 ≈ 36,036
Cost/image    = hourly rate / images per hour
```

| L4 rate | Cost per 1,000 images | Cost per 1,000,000 images |
|---------|-----------------------|---------------------------|
| $0.44/hr | $0.0122 | $12.20 |
| $0.80/hr | $0.0222 | $22.20 |



---

## 9. Project Structure

```
screen-image-detector/
├── predict.py              # inference: score images (ONNX or PT), print class + confidence
├── training.py             # train YOLO classification model
├── export_onnx.py          # export best .pt -> ONNX
├── requirements.txt        # dependencies (Python 3.10)
├── artifacts/
│   ├── model.pt            # production PyTorch model
│   └── model.onnx          # exported ONNX model (default backend)
├── dataset/                # raw collected images (real / screen)
├── roboflow_training_dataset/  # train/val split + augmented data
├── testing_dataset/        # held-out test images
```

---

## 10. Design Questions

### 10.1 How would you keep it accurate as cheaters adapt?
Cheaters will keep trying new ways to fool the model, so we have to keep learning too. Whenever
they find a new trick to get past the detector, we look at what they did, collect those kinds of
images, add them to our dataset, and retrain the model from time to time. This way the model
keeps up with the new tricks instead of falling behind.

### 10.2 How would you make it tiny and fast enough for a phone?
Instead of training a big or "x" model, we train a small or nano model on a large amount of data.
A smaller model has a small weight size and runs fast, so it can easily work on a phone (on the
edge). We also run it with ONNX, which is lightweight and doesn't need a server — so everything
happens right on the device.

### 10.3 How would you choose the cut-off score for flagging fraud?
The threshold is **not something to guess** — it's set through **experimentation**. Using the
real variety of images coming from users, we experiment with different cut-off values and
measure the trade-off between catching fraud (recall) and wrongly flagging genuine photos
(false positives). The threshold is then chosen based on what balance the business needs, and
re-tuned as new user data arrives.
