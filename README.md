# Master's Thesis Results

## The previous installation of detectron2, Mask2Former and detrex is required for inference

## The individual conda envs are given in envs folder:

- base: For all models directly integrated in detectron2
- detrex: For detrex models
- mask2former: For Mask2Former model 
- graph: For BS-HumanGraph

## For inference of detrex models or Mask2Former please use the built in methods. For inference of models directly integrated in detectron2 use the following sh command in the corresponding models subfolder:

```sh 
./inference.sh