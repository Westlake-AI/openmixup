# Installation

In this section we demonstrate how to prepare an environment with PyTorch.

- [Installation](#installation)
  - [Requirements](#requirements)
  - [Install openmixup](#install-openmixup)
  - [Customized installation](#customized-installation)
  - [Prepare datasets](#prepare-datasets)
  - [A from-scratch setup script](#a-from-scratch-setup-script)
  - [Common Issues](#common-issues)

## Requirements

- Linux (Windows is not officially supported)
- Python 3.6+
- PyTorch 1.8 or higher
- CUDA 10.1 or higher
- NCCL 2
- GCC 4.9 or higher
- [mmcv-full](https://github.com/open-mmlab/mmcv) 1.4.7 or higher (use `mmcv` for fast installation)

We have tested the following versions of OS and softwares:

- OS: Ubuntu 16.04/18.04 and CentOS 7.2
- CUDA: 10.0/10.1/11.0/11.2
- NCCL: 2.1.15/2.2.13/2.3.7/2.4.2 (PyTorch-1.1 w/ NCCL-2.4.2 has a deadlock bug, see [here](https://github.com/open-mmlab/OpenSelfSup/issues/6))
- GCC(G++): 4.9/5.3/5.4/7.3/7.4/7.5

## Install openmixup

We recommend that users follow our best practices to install OpenMixup.

**Step 0.** Create a conda virtual environment and activate it.

```shell
conda create -n openmixup python=3.8 -y
conda activate openmixup
```

**Step 1.** Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), e.g., on GPU platforms:

```shell
conda install pytorch torchvision torchaudio cudatoolkit=10.1 -c pytorch
# assuming CUDA=10.1, "pip install torch==1.8.1+cu101 torchvision==0.9.1+cu101 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html"
```

**Step 2.** Install [MMCV](https://github.com/open-mmlab/mmcv) using [MIM](https://github.com/open-mmlab/mim). We recommend the users to install mmcv-full from the source using MIM (or it will install from the source with pip in step 4). You can also use `pip install mmcv` for fast installation.

```shell
pip install -U openmim
mim install mmcv-full
```

**Step 3.** Install other third-party libraries (not necessary). Please install [pyGCO](https://github.com/Borda/pyGCO) for PuzzleMix (used for cut_grid_graph, DON'T USE `pip install gco==1.0.1`).

```shell
conda install faiss-gpu cudatoolkit=10.1 -c pytorch  # optional for DeepCluster and ODC, assuming CUDA=10.1
pip install opencv-contrib-python  # optional for SaliencyMix (cv2.saliency.StaticSaliencyFineGrained_create())
```

**Step 4.** Install OpenMixup. To develop and run openmixup directly, install it from the source:

```shell
git clone https://github.com/Westlake-AI/openmixup.git
cd openmixup
pip install -v -e .
# "-v" means verbose, and "-e" means installing in editable mode;
# or "python setup.py develop"
```

**Step 5.** Install Apex (optional), following the [official instructions](https://github.com/NVIDIA/apex), e.g.
```shell
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
If some errors occur when you install Apex from the source, you can try `python setup.py install` for fast installation. Note that we recommend using PyTorch AMP for mixed precision training in high versions of PyTorch.

**Note:**

1. The git commit id will be written to the version number with step d, e.g. 0.1.0+2e7045c. The version will also be saved in trained models.

2. Following the above instructions, openmixup is installed on `dev` (editable) mode, and any local modifications made to the code will take effect immediately (except for the running experiments). You can install it to `pip/conda` by `pip install .` and the local modifications will not take effect without reinstalling it.

3. If you are installing `cv2` for the first time, `ImportError: libGL.so.1` will occur, which can be solved by `apt install libgl1-mesa-glx`. If you would like to use `opencv-python-headless` instead of `opencv-python`, you can install it before installing MMCV. Refer to [issue #48](https://github.com/Westlake-AI/openmixup/issues/48) for some errors encountered with the version of `cv2`.

4. Some errors with mmcv installation can be solved according to the issue of [MMCV](https://github.com/open-mmlab/mmcv), e.g., using `yapf<=0.40.1` for [issue #10962](https://github.com/open-mmlab/mmdetection/issues/10962).

<p align="right">(<a href="#top">back to top</a>)</p>

## Customized installation

### Benchmark

According to [MMSelfSup](https://github.com/open-mmlab/mmselfsup), if you need to evaluate your pre-training model with some downstream tasks such as detection or segmentation, please also install [Detectron2](https://github.com/facebookresearch/detectron2), [MMDetection](https://github.com/open-mmlab/mmdetection) and [MMSegmentation](https://github.com/open-mmlab/mmsegmentation).

If you don't run MMDetection and MMSegmentation benchmark, it is unnecessary to install them.

You can simply install MMDetection and MMSegmentation with the following command:

```shell
pip install mmdet mmsegmentation
```

For more details, you can check the installation page of [MMDetection](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/get_started.md) and [MMSegmentation](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/get_started.md).

### Install MMCV without MIM

MMCV contains C++ and CUDA extensions, thus depending on PyTorch in a complex way. MIM solves such dependencies automatically and makes the installation easier. However, it is not a must. To install MMCV with pip instead of MIM, please follow [MMCV installation guides](https://mmcv.readthedocs.io/en/latest/get_started/installation.html). This requires manually specifying a find-url based on PyTorch version and its CUDA version.

For example, the following command install mmcv-full built for PyTorch 1.10.x and CUDA 11.3.

```shell
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10/index.html
```

### Another option: Docker Image

We provide a [Dockerfile](/docker/Dockerfile) to build an image.

```shell
# build an image with PyTorch 1.10.0, CUDA 11.3, CUDNN 8.
docker build -f ./docker/Dockerfile --rm -t openmixup:torch1.10.0-cuda11.3-cudnn8 .
```

**Note:** Make sure you've installed the [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker).


## Prepare datasets

It is recommended to symlink your dataset root (assuming `$YOUR_DATA_ROOT`) to `$OPENMIXUP/data`.
If your folder structure is different, you may need to change the corresponding paths in config files.

### Prepare Classification Datasets

We support following datasets: CIFAR-10/100, [Tiny-ImageNet](https://www.kaggle.com/c/tiny-imagenet), [ImageNet-1k](http://www.image-net.org/challenges/LSVRC/2012/), [Place205](http://places.csail.mit.edu/downloadData.html), [iNaturalist2017/2018](https://github.com/visipedia/inat_comp), [CUB200](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html), [FGVC-Aircrafts](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/), [StandordCars](http://ai.stanford.edu/~jkrause/cars/car_dataset.html). Taking ImageNet for example, you need to 1) download ImageNet; 2) create the following list files or download [meta files](https://github.com/Westlake-AI/openmixup/releases/download/dataset/meta.zip) under $DATA/meta/: `train.txt` and `val.txt` contains an image file name in each line, `train_labeled.txt` and `val_labeled.txt` contains `filename label\n` in each line; `train_labeled_*percent.txt` are the down-sampled lists for semi-supervised evaluation. 3) create a symlink under `$OPENMIXUP/data/`.

### Prepare PASCAL VOC

Assuming that you usually store datasets in `$YOUR_DATA_ROOT` (e.g., for me, `/home/xhzhan/data/`).
This script will automatically download PASCAL VOC 2007 into `$YOUR_DATA_ROOT`, prepare the required files, create a folder `data` under `$OPENSELFSUP` and make a symlink `VOCdevkit`.

```shell
cd $OPENMIXUP
bash tools/prepare_data/prepare_voc07_cls.sh $YOUR_DATA_ROOT
```

At last, the folder with all related datasets looks like:

```
openmixup
├── openmixup
├── benchmarks
├── configs
├── data
│   ├── meta [used for 'ImageList' dataset]
│   ├── ade
│   ├── cifar10
│   ├── cifar100
│   │   ├── cifar-100-batches-py
│   │   ├── cifar-100-python.tar
│   │── coco
│   │── CUB200
│   ├── FGVC_Aircrafts
│   │   |   ├── images (contains all train & val)
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
│   │── StanfordCars
│   │   ├── test
│   │   ├── train
│   │── STL10
│   │   ├── test
│   │   ├── train
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

## A from-scratch setup script

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

<p align="right">(<a href="#top">back to top</a>)</p>

## Common Issues

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

### Issues of bugs

1. PyTorch-1.8 has a bug in the AdamW optimizer, which will cause some errors in DDP training. See this [issue](https://github.com/pytorch/pytorch/pull/52944).
2. PyTorch-1.8 or higher has a bug in printing logs to the console. The log and log.json files are not affected.
3. The training hangs / deadlocks in some intermediate iterations. See this [issue](https://github.com/open-mmlab/OpenSelfSup/issues/6). This bug is fixed in the higher versions of PyTorch>=1.6.

<p align="right">(<a href="#top">back to top</a>)</p>
