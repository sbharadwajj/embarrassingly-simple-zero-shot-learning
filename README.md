# Embarrsingly simple zero-shot learning

This is the implementation of the paper "An embarrassingly simple approach to zero-shot learning." (EsZsl) ICML, [[pdf]](http://proceedings.mlr.press/v37/romera-paredes15.pdf). 

The file `demo_eszsl` is a jupyter notebook which contains a walk through of EsZsl.

# Dataset

The dataset splits can be downloaded [here](https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/zero-shot-learning/zero-shot-learning-the-good-the-bad-and-the-ugly/), please download the `Proposed Split` and place it in the same folder. 

Find additional details about the dataset in the `README.md` of the `Proposed split`.

# Training and Testing

If you want to skip the demo and just run training and testing for different dataset splits use:

```
python main.py --dataset SUN --dataset_path xlsa17/data/ --alpha 3 --gamma 1

```
Setting the hyperparameters alpha and gamma is optional. If the values are not given, the code will evaluate on the train and validation set to find the best hyperparameters.

# Results

This version does not have the kernel implementation used in the paper. Hence the results fluctuate by a margin of 1-4%. 

The results are taken from the paper [Zero-Shot Learning - A Comprehensive Evaluation of the Good, the Bad and the Ugly](https://arxiv.org/pdf/1707.00600.pdf) and are evaluated for features extracted from ResNet-50 for the Proposed split.

| Dataset       | Paper - (top-1 accuracy in %) | Respository Results | Hyper-params(trainval & test) |
| ------------- |:-----------------------------:| -------------------:| --------------------------:|
| CUB        |    53.9      | 51.31 | Alpha=2, Gamma=0 |
| AWA1    |  58.2 | 56.19 | Alpha=3, Gamma=0 |
| AWA2 | 58.6 | 54.50 | Alpha=3, Gamma=0 |
| aPY | 38.3 | 38.47 | Alpha=3, Gamma=-1|
| SUN | 54.5 | 55.62 | Alpha=2, Gamma=2 |


# Refrences

If this repository was useful for your research, please cite.

@misc{chichilicious,
  author = {Bharadwaj, Shrisha},
  title = {embarrsingly-simple-zero-shot-learning},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/chichilicious/embarrsingly-simple-zero-shot-learning}},
}