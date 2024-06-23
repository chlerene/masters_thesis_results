#!/usr/bin/env bash

import detectron2
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

from detectron2.projects import point_rend

import numpy as np
import torch
import os

register_coco_instances("atlas_val_segm", {},"datasets/Atlas/approach_1/labels_val.json", "datasets/Atlas/images/data_val/")

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 11
cfg.MODEL.WEIGHTS = "models/anatomy_segmentation/approach_1/MaskRCNN/model_final.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.0
predictor = DefaultPredictor(cfg)

evaluator = COCOEvaluator("atlas_val_segm", output_dir="./inference")
val_loader = build_detection_test_loader(cfg, "atlas_val_segm")
print(inference_on_dataset(predictor.model, val_loader, evaluator))
