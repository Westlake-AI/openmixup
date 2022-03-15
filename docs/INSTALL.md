## Installation

### Requirements

- Linux (Windows is not officially supported)
- Python 3.6+
- PyTorch 1.6 or higher
- CUDA 10.0 or higher
- NCCL 2
- GCC 4.9 or higher
- [mmcv](https://github.com/open-mmlab/mmcv) 1.2.7 or higher

We have tested the following versions of OS and softwares:

- OS: Ubuntu 16.04/18.04 and CentOS 7.2
- CUDA: 10.0/10.1/11.0/11.2
- NCCL: 2.1.15/2.2.13/2.3.7/2.4.2 (PyTorch-1.1 w/ NCCL-2.4.2 has a deadlock bug, see [here](https://github.com/open-mmlab/OpenSelfSup/issues/6))
- GCC(G++): 4.9/5.3/5.4/7.3/7.4/7.5

### Install openmixup

a. Create a conda virtual environment and activate it.

```shell
conda create -n openmixup python=3.8 -y
conda activate openmixup
```

b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), e.g.,

```shell
conda install pytorch torchvision torchaudio cudatoolkit=10.1 -c pytorch
# assuming CUDA=10.1, "pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html"
```

c. Install other third-party libraries (not necessary). Please install [pyGCO](https://github.com/Borda/pyGCO) for PuzzleMix (used for cut_grid_graph, DON'T USE `pip install gco==1.0.1`).

```shell
conda install faiss-gpu cudatoolkit=10.1 -c pytorch  # optional for DeepCluster and ODC, assuming CUDA=10.1
pip install opencv-contrib-python  # optional for SaliencyMix (cv2.saliency.StaticSaliencyFineGrained_create())
```

d. Clone the openselfsup repository.

```shell
git clone https://github.com/Westlake-AI/openmixup.git
cd openmixup
```

e. Install.

```shell
pip install -v -e .  # or "python setup.py develop"
```

f. Install Apex (optional), following the [official instructions](https://github.com/NVIDIA/apex), e.g.
```shell
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

Note:

1. The git commit id will be written to the version number with step d, e.g. 0.1.0+2e7045c. The version will also be saved in trained models.

2. Following the above instructions, openselfsup is installed on `dev` mode, any local modifications made to the code will take effect without the need to reinstall it (unless you submit some commits and want to update the version number).

3. If you would like to use `opencv-python-headless` instead of `opencv-python`,
you can install it before installing MMCV.


### Prepare datasets

It is recommended to symlink your dataset root (assuming $YOUR_DATA_ROOT) to `$OPENMIXUP/data`.
If your folder structure is different, you may need to change the corresponding paths in config files.

#### Prepare Classification Datasets

We support following datasets: CIFAR-10/100, [Tiny-ImageNet](https://www.kaggle.com/c/tiny-imagenet), [ImageNet-1k](http://www.image-net.org/challenges/LSVRC/2012/), [Place205](http://places.csail.mit.edu/downloadData.html), [iNaturalist2017/2018](https://github.com/visipedia/inat_comp), [CUB200](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html), [FGVC-Aircrafts](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/), [StandordCars](http://ai.stanford.edu/~jkrause/cars/car_dataset.html). Taking ImageNet for example, you need to 1) download ImageNet; 2) create the following list files or download [meta](https://github.com/Westlake-AI/openmixup/releases/download/dataset/meta.zip) under $DATA/meta/: `train.txt` and `val.txt` contains an image file name in each line, `train_labeled.txt` and `val_labeled.txt` contains `filename[space]label\n` in each line; `train_labeled_*percent.txt` are the down-sampled lists for semi-supervised evaluation. 3) create a symlink under `$OPENMIXUP/data/`.

#### Prepare PASCAL VOC

Assuming that you usually store datasets in `$YOUR_DATA_ROOT` (e.g., for me, `/home/xhzhan/data/`).
This script will automatically download PASCAL VOC 2007 into `$YOUR_DATA_ROOT`, prepare the required files, create a folder `data` under `$OPENSELFSUP` and make a symlink `VOCdevkit`.

```shell
cd $OPENMIXUP
bash tools/prepare_data/prepare_voc07_cls.sh $YOUR_DATA_ROOT
```

At last, the folder looks like:

```
OpenSelfSup
├── openmixup
├── benchmarks
├── configs
├── data
│   ├── meta [used for 'ImageList' dataset]
│   ├── Aircrafts
│   │   |   ├── images (contains all train & val)
│   ├── cifar10
│   ├── cifar100
│   │   ├── cifar-10-batches-py
│   │   ├── cifar-10-python.tar
│   │── Cars
│   │── CUB200
│   ├── ImageNet
│   │   ├── train
│   │   |   ├── n01440764
│   │   |   ├── n01443537
│   │   |   ...
│   │   |   ├── n15075141
│   │   ├── val
│   │── iNaturalist2017
│   │── iNaturalist2018
│   ├── Places205
│   │   ├── images256
│   │   |   ├── a
│   │   |   |   ├── abbey
│   │   |   |   ├── airport_terminal
│   │   |   |   ...
│   │   |   ├── b
│   │   |   ...
│   │   |   ├── y
│   │── STL10
│   ├── TinyImageNet
│   │   ├── train
│   │   |   ├── n01443537
│   │   |   ...
│   │   ├── val
│   │   |   ├── images (contains all train & val)
│   ├── VOCdevkit
│   │   ├── VOC2007
│   │   ├── VOC2012
```

### A from-scratch setup script

Here is a full script for setting up openmixup with conda and link the dataset path. The script does not download full datasets, you have to prepare them on your own.

```shell
conda create -n openmixup python=3.8 -y
conda activate openmixup

conda install -c pytorch pytorch torchvision -y
git clone https://github.com/Westlake-AI/openmixup.git
cd openmixup
python setup.py develop

# download 'meta' and move to data/meta
wget https://github.com/Westlake-AI/openmixup/releases/download/dataset/meta.zip
unzip -d data/meta meta.zip
# download full classification datasets
ln -s $CIFAR10_ROOT data/cifar10
ln -s $CIFAR100_ROOT data/cifar100
ln -s $IMAGENET_ROOT data/ImageNet
ln -s $TINY_ROOT data/TinyImagenet
# download VOC datasets
bash tools/prepare_data/prepare_voc07_cls.sh $YOUR_DATA_ROOT
```

### Using multiple openmixup versions

If there are more than one `openmixup` on your machine, and you want to use them alternatively, the recommended way is to create multiple conda environments and use different environments for different versions. The `develop` mode is recommanded if you want to add your own codes in `openmixup`.

Another way is to insert the following code to the main scripts (`train.py`, `test.py` or any other scripts you run)
```python
import os.path as osp
import sys
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))
```

Or run the following command in the terminal of corresponding folder to temporally use the current one.
```shell
export PYTHONPATH=`pwd`:$PYTHONPATH
```

## Common Issues

1. The training hangs / deadlocks in some intermediate iteration. See this [issue](https://github.com/open-mmlab/OpenSelfSup/issues/6). We haven't found this issue when using higher versions of PyTorch.
