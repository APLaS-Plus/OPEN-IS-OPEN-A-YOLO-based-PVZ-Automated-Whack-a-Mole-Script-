from ultralytics import YOLO
import os

ROOTPATH = os.path.dirname(os.path.abspath(__file__))

model = YOLO("yolo11s.yaml")

model.train(
    data=os.path.join(ROOTPATH, "configs", "ZVP.yaml"),
    cfg=os.path.join(ROOTPATH, "configs", "trainCfg.yaml"),
)
