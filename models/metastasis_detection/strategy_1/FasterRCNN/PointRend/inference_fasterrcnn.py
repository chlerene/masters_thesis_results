import detectron2
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

import numpy as np
import torch
import os
import cv2

register_coco_instances("bs-80k_val_det", {},"datasets/BS-80K/strategy_1/labels_val.json", "datasets/BS-80K/images/data_val/")

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = "models/metastasis_detection/strategy_1/FasterRCNN/PointRend/model_final.pth"

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.0
predictor = DefaultPredictor(cfg)

evaluator = COCOEvaluator("bs-80k_val_det", output_dir="./inference")
val_loader = build_detection_test_loader(cfg, "bs-80k_val_det")
print(inference_on_dataset(predictor.model, val_loader, evaluator))

