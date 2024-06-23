#!/bin/bash

echo 'Start inference'
srun --partition GPUampere --gpus 1 --time=01:00:00 python3 inference_maskrcnn.py
