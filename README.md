# Deep-STORM_PyTorch
Port of Deep-STORM from Tensorflow to PyTorch

[Original Repository](https://github.com/EliasNehme/Deep-STORM)

## PyTorch Pre-Trained Files
| [Deep-STORM PyTorch Weights](https://drive.google.com/file/d/12LzHf2NC9vuhq3mYALGHyI7-glW0Nfoy/view?usp=sharing) | [Deep-STORM Mean and Standard Deviation](https://drive.google.com/file/d/1GDG8_-qbKdTpMltz03jzmH8FMJ7ZgD9L/view?usp=sharing) |

## Python/Conda
This project was implemented using Python 3.9.5 and PyTorch 1.8.0.

I suggest you use anaconda/miniconda/miniforge to setup a python enviroment.

```sh
conda create -n torch python=3.9.5 
conda activate torch
conda install pytorch=1.8.0 torchvision torchaudio -c pytorch  # modify based on GPU requirements
```

The remaining dependencies can also be installed with conda
```sh
conda install -c conda-forge pip matplotlib scikit-image scikit-learn tqdm h5py scipy numpy
```
