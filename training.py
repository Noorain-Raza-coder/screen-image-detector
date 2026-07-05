from ultralytics import YOLO

TRAINING_DATASET_PATH = "roboflow_downloaded_dataset"  # path to the training dataset

model = YOLO("yolo26s-cls.pt")  # or use a custom model path
model.train(
                data=TRAINING_DATASET_PATH,
                epochs=250,
                batch=16,
                patience=25,
                imgsz=720,
                lr0=0.001,
                save=True,
                device=0,
                exist_ok=True)