import os
import torch
from ultralytics import YOLO

def main():
    base_path = "/home/jooyeon/ros2_ws/src/doosan-robot2/dataset"
    data_yaml_path = os.path.join(base_path, "data.yaml")

    device_env = 0 if torch.cuda.is_available() else "cpu"

    model = YOLO("yolov8n-seg.pt")
    model.train(
        data=data_yaml_path,
        epochs=50,
        imgsz=640,
        device=device_env,
    )

if __name__ == "__main__":
    main()