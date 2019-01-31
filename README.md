# Embarrsingly simple zero-shot learning

This is the implementation of the paper "An embarrassingly simple approach to zero-shot learning." (EsZsl) ICML, [[pdf]](http://proceedings.mlr.press/v37/romera-paredes15.pdf). The file `demo_eszsl` is a jupyter notebook which contains a walk through of EsZsl.

# Dataset

The dataset splits can be downloaded [here](https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/zero-shot-learning/zero-shot-learning-the-good-the-bad-and-the-ugly/), please download the `Proposed Split` and place it in the same folder. 

The dataset is structured as follows:\
The proposed splits consist of:
- allclasses.txt: list of names of all classes in the dataset
- trainvalclasses.txt: seen classes
- testclasses.txt: unseen classes
- trainclasses1/2/3.txt: 3 different subsets of trainvalclasses used for tuning the hyperparameters 
- valclasses1/2/3.txt: 3 different subsets of trainvalclasses used for tuning the hyperparameters


resNet101.mat includes the following fields:
- features: columns correspond to image instances
- labels: label number of a class is its row number in allclasses.txt
- image_files: image sources  


att_splits.mat includes the following fields:
- att: columns correpond to class attribute vectors normalized to have unit l2 norm, following the classes order in allclasses.txt 
- original_att: the original class attribute vectors without normalization
- trainval_loc: instances indexes of train+val set features (for only seen classes) in resNet101.mat
- test_seen_loc: instances indexes of test set features for seen classes
- test_unseen_loc: instances indexes of test set features for unseen classes
- train_loc: instances indexes of train set features (subset of trainval_loc)
- val_loc: instances indexes of val set features (subset of trainval_loc)

