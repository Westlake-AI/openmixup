# Getting Started

- [Getting Started](#getting-started)
  - [Introduction](#introduction)
  - [Train existing methods](#train-existing-methods)
    - [Train with single/multiple GPUs](#train-with-singlemultiple-gpus)
    - [Train with multiple machines](#train-with-multiple-machines)
    - [Launch multiple jobs on a single machine](#launch-multiple-jobs-on-a-single-machine)
    - [Gradient Accumulation](#gradient-accumulation)
    - [Mixed Precision Training](#mixed-precision-training)
    - [Speeding Up IO](#speeding-up-io)
  - [Benchmarks](#benchmarks)
  - [Tools and Tips](#tools-and-tips)
    - [Generate fast config files](#generate-fast-config-files)
    - [Count number of parameters](#count-number-of-parameters)
    - [Publish a model](#publish-a-model)
    - [Reproducibility](#reproducibility)
    - [Convenient Features](#convenient-features)

This page provides basic tutorials about the usage of OpenMixup. For installation instructions, please see [Install](docs/en/install.md).

## Introduction

Learning discriminative visual representation efficiently that facilitates downstream tasks is one of the fundamental problems in computer vision. Data mixing techniques largely improve the quality of deep neural networks (DNNs) in various scenarios. Since mixup techniques are used as augmentations or auxiliary tasks in a wide range of cases, this repo focuses on mixup-related methods for Supervised, Self- and Semi-Supervised Representation Learning. Thus, we name this repo `OpenMixp`.

## Train existing methods

**Note**: The default learning rate in config files is for 4 or 8 GPUs. If using differnt number GPUs, the total batch size will change in proportion, you have to scale the learning rate following `new_lr = old_lr * new_ngpus / old_ngpus`. We recommend to use `tools/dist_train.sh` even with 1 gpu, since some methods do not support non-distributed training.

### Train with single/multiple GPUs

```shell
bash tools/dist_train.sh ${CONFIG_FILE} ${GPUS} [optional arguments]
```
Optional arguments are:
- `--resume_from ${CHECKPOINT_FILE}`: Resume from a previous checkpoint file.
- `--pretrained ${PRETRAIN_WEIGHTS}`: Load pretrained weights for the backbone.
- `--deterministic`: Switch on "deterministic" mode which slows down training but the results are reproducible.

An example: Run the following command, training results (checkpoints, jsons, logs) saved in `WORK_DIR=work_dirs/classification/imagenet/mixups/basic/r50/mix_modevanilla/r50_mixups_CE_none_ep100/`.
```shell
bash tools/dist_train.sh configs/classification/imagenet/mixups/basic/r50/mix_modevanilla/r50_mixups_CE_none_ep100.py 8
```
**Note**: During training, checkpoints and logs are saved in the same folder structure as the config file under `work_dirs/`. Custom work directory is not recommended since evaluation scripts infer work directories from the config file name. If you want to save your weights somewhere else, please use symlink, for example:

```shell
ln -s /lisiyuan/source/OPENMIXUP_WORKDIRS ${OPENMIXUP}/work_dirs
```

Alternatively, if you run OpenMixup on a cluster managed with [slurm](https://slurm.schedmd.com/):
```shell
SRUN_ARGS="${SRUN_ARGS}" bash tools/srun_train.sh ${PARTITION} ${CONFIG_FILE} ${GPUS} [optional arguments]
```

An example:
```shell
SRUN_ARGS="-w xx.xx.xx.xx" bash tools/srun_train.sh Dummy configs/selfsup/odc/r50_v1.py 8 --resume_from work_dirs/selfsup/odc/r50_v1/epoch_100.pth
```

### Train with multiple machines

If you launch with multiple machines simply connected with ethernet, you have to modify `tools/dist_train.sh` or create a new script, please refer to PyTorch [Launch utility](https://pytorch.org/docs/stable/distributed.html#launch-utility). Usually it is slow if you do not have high speed networking like InfiniBand.

If you launch with slurm, the command is the same as that on single machine described above. You only need to change ${GPUS}, e.g., to 16 for two 8-GPU machines.

### Launch multiple jobs on a single machine

If you launch multiple jobs on a single machine, e.g., 2 jobs of 4-GPU training on a machine with 8 GPUs,
you need to specify different ports (29500 by default) for each job to avoid communication conflict.

If you use `dist_train.sh` to launch training jobs:
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 bash tools/dist_train.sh ${CONFIG_FILE} 4
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29501 bash tools/dist_train.sh ${CONFIG_FILE} 4
```
For example, you can run the script below to train a mixup CIFAR100 classification algorithm with 4 GPUs:
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 bash tools/dist_train.sh openmixup\configs\classification\cifar100\mixups\basic\r18_mixups_CE_none.py 4
```

If you use launch training jobs with slurm:
```shell
GPUS_PER_NODE=4 bash tools/srun_train.sh ${PARTITION} ${CONFIG_FILE} 4 --port 29500
GPUS_PER_NODE=4 bash tools/srun_train.sh ${PARTITION} ${CONFIG_FILE} 4 --port 29501
```

### Gradient Accumulation

If you do not have so many GPUs to launch large training jobs, we recommend the gradient accumulation. Assuming that you only have 1 GPU that can contain 64 images in a batch, while you expect the batch size to be 256, you may add the following line into your config file. It performs network update every 4 iterations. In this way, the equivalent batch size is 256. Of course, it is about 4x slower than using 4 GPUs. Note that the workaround is not applicable for methods like SimCLR which require intra-batch communication.

```python
optimizer_config = dict(update_interval=4)
```

### Mixed Precision Training

We support [mmcv](https://github.com/open-mmlab/mmcv) and [Apex](https://github.com/NVIDIA/apex) to implement Mixed Precision Training. If you want to use Mixed Precision Training, you can add below in the config file.
```python
use_fp16 = True
fp16 = dict(type='mmcv', loss_scale='dynamic')
```
You can choose FP16 types in 'apex' or 'mmcv'. We recommend that using 'mmcv' in **PyTorch 1.6** or higher for faster training speed, while using 'apex' with lower PyTorch versions. An example of the RSB A3 setting:
```python
bash tools/dist_train.sh configs/classification/imagenet/mixups/rsb_a3/r50/r18_rsb_a3_CE_sigm_mix0_1_cut1_0_sz160_bs2048_fp16_ep100.py 4
```

### Speeding Up IO
1 . Prefetching data helps to speeding up IO and make better use of CUDA stream parallelization. If you want to use it, you can activate it in the config file (disabled by default) and remove `ToTensor` and `Normalize` in 'train_pipeline'. Costly operation `ToTensor` is reimplemented along with prefetch.
```python
prefetch = True
if not prefetch:
    train_pipeline.extend([dict(type='ToTensor'), dict(type='Normalize', **img_norm_cfg)])
```

2 . Replacing Pillow with Pillow-SIMD (https://github.com/uploadcare/pillow-simd.git) to make use of SIMD command sets with modern CPU.
 ```shell
pip uninstall pillow
pip install Pillow-SIMD or CC="cc -mavx2" pip install -U --force-reinstall pillow-simd if AVX2 is available.
```
We test it using MoCoV2 using a total batch size of 256 on Tesla V100. The training time per step is decreased to 0.17s from 0.23s.

## Benchmarks

We provide several standard benchmarks to evaluate representation learning (supervised and self-supervised pre-trained models), and you can refer to [Benchmarks](./tutorials/6_benchmarks.md) for the details. The config files or scripts for evaluation mentioned are NOT recommended to be changed if you want to use this repo in your publications. We hope that all methods are under a fair comparison.

## Tools and Tips

### Generate fast config files

If you want to adjust some parts of a basic config file (e.g., do ablation studies or tuning hyper-parameters), we provide ConfigGenerator in the config folders of each methods. For example, you want to train {'Mixup', 'CutMix'} with alpha in {0.2, 1.0} for {100, 300} epochs on ImageNet-1k based on PyTorch-style settings in `configs/classification/imagenet/mixups/basic/r50_mixups_CE_none.py`, you can modified `auto_train_in_mixups.py` and run
```python
python configs/classification/imagenet/mixups/auto_train_in_mixups.py
```
It will generate eight config files and a bash file `r50_mixups_CE_none_xxxx.sh`. You can adjust GPUs and PORT settings and execute this bash file to run eight experiments automaticly.

### Count number of parameters

```shell
python tools/count_parameters.py ${CONFIG_FILE}
```

### Publish a model

Compute the hash of the weight file and append the hash id to the filename. The output file is the input file name with a hash suffix.

```shell
python tools/publish_model.py ${WEIGHT_FILE}
```
Arguments:
- `WEIGHT_FILE`: The extracted backbone weights extracted aforementioned.

### Reproducibility

If you want to make your performance exactly reproducible, please switch on `--deterministic` to train the final model to be published. Note that this flag will switch off `torch.backends.cudnn.benchmark` and slow down the training speed.

### Convenient Features

* Configure data augmentations in the config file.

The augmentations are the same as `torchvision.transforms` except that `torchvision.transforms.RandomAppy` corresponds to `RandomAppliedTrans`. `Lighting` and `GaussianBlur` is additionally implemented.

```python
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [
    dict(type='RandomResizedCrop', size=224),
    dict(type='RandomAppliedTrans',
        transforms=[
            dict(type='GaussianBlur', sigma_min=0.1, sigma_max=2.0, kernel_size=23)],
        p=0.5),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg)
]
```

* Parameter-wise optimization parameters.

You may specify optimization paramters including lr, momentum and weight_decay for a certain group of paramters in the config file with `paramwise_options`. `paramwise_options` is a dict whose key is regular expressions and value is options. Options include 6 fields: lr, lr_mult, momentum, momentum_mult, weight_decay, weight_decay_mult, lars_exclude (only works with LARS optimizer).

```python
# this config sets all normalization layers in CNN with weight_decay_mult=0.1,
# and the `head` with `lr_mult=10, momentum=0`.
paramwise_options = {
    '(bn|gn)(\d+)?.(weight|bias)': dict(weight_decay_mult=0.1),
    '\Ahead.': dict(lr_mult=10, momentum=0)}
optimizer_cfg = dict(type='SGD', lr=0.01, momentum=0.9,
                     weight_decay=0.0001,
                     paramwise_options=paramwise_options)
```

* Configure custom hooks in the config file.

The hooks will be called in order. For hook design, please refer to [momentum_hook.py](https://github.com/Westlake-AI/openmixup/blob/main/openmixup/hooks/momentum_hook.py) as an example.

```python
custom_hooks = [
    dict(type='SAVEHook', ...),
    dict(type='CosineScheduleHook', ...),
]
```
