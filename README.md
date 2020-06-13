## Brain Image Analysis using Deep Learning
A multimodal deep learning approach to the early prediction of mild cognitive impairment conversion to Alzheimer's Disease.
## Overview
This project contains the code for the convolutional neural network - MudNet that identifies AD-converters and their risk of conversion within a 24 months.
* ```train.py``` defines that model training and its parameters
* ```train_crossval.py``` is the cross-validation implementation for evaluating the model
* ```utils``` folder contains:
  * ```callbacks.py``` extracts model metrics
  * ```data_loader.py``` loads and prepares structural MRI and clinical data
  * ```model.py``` defines MudNet layers and parameters
  * ```preprocess.py``` contains data splitting and pre-processing methods (i.e. train-test split, one-hot encoding etc)
## Model Architecture
<br></br>
<img src="figures/ArchitectureOverview.png" width="656" height="1165">
<br></br>
<img src="figures/ArchitectureLayers.png" width="640" height="496">
