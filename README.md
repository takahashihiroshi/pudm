# Positive-Unlabeled Diffusion Models for Preventing Sensitive Data Generation
This is a pytorch implementation of the following paper [[openreview]](https://openreview.net/forum?id=jKcZ4hF4s5):
```
@inproceedings{pudm,
  title={Positive-Unlabeled Diffusion Models for Preventing Sensitive Data Generation},
  author={Takahashi, Hiroshi and Iwata, Tomoharu and Kumagai, Atsutoshi and Yamanaka, Yuuki and Yamashita, Tomoya},
  booktitle={The Thirteenth International Conference on Learning Representations}
}
```
Please read [LICENCE.md](LICENCE.md) before reading or using the files.


## Prerequisites
- Please see [requirements.txt](requirements.txt)


## Datasets
- MNIST, CIFAR10, STL10, and CelebA will be downloaded when first used.
- For Stable Diffusion experiments, please generate images of Brad Pitt and middle-aged men using Stable Diffusion and organize them into folders as follows.
```
datasets/middle_aged_man100
├── 0_unlabeled
└── 1_positive
```
- The `0_unlabeled` folder contains 64 images of middle-aged men and 16 images of Brad Pitt.
- The `1_positive` folder contains 20 images of Brad Pitt.
- The prompts used for generation are as follows:
  - Brad Pitt: `a photo of Brad Pitt`
  - Middle-aged man: `a photo of a middle-aged man`


## Usage
- Please run the following scripts:
  - `main.py`: for from-scratch training
  - `main-ft.py`: for fine-tuning pre-trained models
  - `main-sd.py`: for fine-tuning stable diffusion models


### for from-scratch training
```
usage: main.py [-h] [--config_name CONFIG_NAME] [--algorithm ALGORITHM]
               [--beta BETA] [--seed SEED]
```
- You can choose the `config_name` from following configurations: 
  - `MNIST_even`: MNIST dataset where even numbers are normal
  - `MNIST_odd`: MNIST dataset where odd numbers are normal
  - `CIFAR10_vehicles`: CIFAR10 dataset where vehicle images are normal
  - `CIFAR10_animals`: CIFAR10 dataset where animal images are normal
  - `STL10_vehicles`: STL10 dataset where vehicle images are normal
  - `STL10_animals`: STL10 dataset where animal images are normal
- You can choose the `algorithm` from following algorithms:
  - `Unsupervised`: Unsupervised diffusion models
  - `Supervised`: Supervised Diffusion models
  - `PU`: Proposed method
- You can change the `beta`, the hyperparameter of Proposed method
- You can change the random `seed` of the training


### for fine-tuning pre-trained models
```
usage: main-ft.py [-h] [--config_name CONFIG_NAME] [--algorithm ALGORITHM]
                  [--beta BETA] [--seed SEED]
```
- You can choose the `config_name` from following configurations: 
  - `CIFAR10_vehicles`: CIFAR10 dataset where vehicle images are normal
  - `CIFAR10_animals`: CIFAR10 dataset where animal images are normal
  - `CelebA_male`: CelebA dataset where male images are normal
  - `CelebA_female`: CelebA dataset where female images are normal
- You can choose the `algorithm` from following algorithms:
  - `Unsupervised`: Unsupervised diffusion models
  - `Supervised`: Supervised Diffusion models
  - `PU`: Proposed method
- You can change the `beta`, the hyperparameter of Proposed method
- You can change the random `seed` of the training


### for fine-tuning stable diffusion models
```
usage: main-sd.py [-h] [--algorithm ALGORITHM] [--beta BETA] [--seed SEED]
```
- You can choose the `algorithm` from following algorithms:
  - `Unsupervised`: Unsupervised diffusion models
  - `Supervised`: Supervised Diffusion models
  - `PU`: Proposed method
- You can change the `beta`, the hyperparameter of Proposed method
- You can change the random `seed` of the training


## Example
Running `MNIST_even` experiment (normal: even / sensitive: odd) with our approach:
```
python main.py --config_name MNIST_even --algorithm PU --beta 0.1 --seed 42
```
