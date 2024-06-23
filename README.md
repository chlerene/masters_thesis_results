# Master's Thesis Results

## The previous installation of detectron2, Mask2Former and detrex is required for inference

## The individual conda envs are given in envs folder:

- base: For all models directly integrated in detectron2
- detrex: For detrex models
- mask2former: For Mask2Former model 

## For inference of detrex models or Mask2Former model please use their built in methods. For inference of models directly integrated in detectron2 use the following command in the corresponding models subfolder:

```sh 
./inference.sh