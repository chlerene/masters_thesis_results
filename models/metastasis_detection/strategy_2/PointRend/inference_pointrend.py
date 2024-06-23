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

register_coco_instances("bs-80k_val_segm_lesion", {},"datasets/BS-80K/strategy_2/labels_val.json", "datasets/BS-80K/images/data_val/")

cfg = get_cfg()
point_rend.add_pointrend_config(cfg)
cfg.merge_from_file("detectron2/projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco.yaml")
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
cfg.MODEL.POINT_HEAD.NUM_CLASSES = 2
cfg.MODEL.WEIGHTS = "models/metastasis_detection/strategy_2/PointRend/model_final.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0
predictor = DefaultPredictor(cfg)

evaluator = COCOEvaluator("bs-80k_val_segm_lesion", output_dir="./inference")
val_loader = build_detection_test_loader(cfg, "bs-80k_val_segm_lesion")
print(inference_on_dataset(predictor.model, val_loader, evaluator))