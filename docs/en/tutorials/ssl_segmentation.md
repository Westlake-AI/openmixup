# Segmentation

- [Segmentation](#segmentation)
  - [Train](#train)
  - [Test](#test)

For semantic segmentation task, we use MMSegmentation. First, make sure you have installed [MIM](https://github.com/open-mmlab/mim), which is also a project of OpenMMLab.

```shell
pip install openmim
mim install mmsegmentation
```

It is very easy to install the package.

Besides, please refer to MMSegmentation for [installation](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/get_started.md) and [data preparation](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/dataset_prepare.md#prepare-datasets).

## Train

After installation, you can run MMSeg with simple command.

```shell
# distributed version
bash benchmarks/mmsegmentation/mim_dist_train.sh ${CONFIG} ${PRETRAIN} ${GPUS}

# slurm version
bash benchmarks/mmsegmentation/mim_slurm_train.sh ${PARTITION} ${CONFIG} ${PRETRAIN}
```

Remarks:

- `CONFIG`: Use config files under `configs/benchmarks/mmsegmentation/` or write your own config files
- `PRETRAIN`: the pre-trained model file (the backbone parameters only).
- `${GPUS}`: The number of GPUs that you want to use to train. We adopt 4 GPUs for segmentation tasks by default.

Example:

```shell
bash benchmarks/mmsegmentation/mim_dist_train.sh \
configs/benchmarks/mmsegmentation/voc12aug/fcn_r50-d8_4xb4-20k_voc12aug-512x512.py \
https://download.openmmlab.com/mmselfsup/1.x/byol/byol_resnet50_16xb256-coslr-200e_in1k/byol_resnet50_16xb256-coslr-200e_in1k_20220825-de817331.pth 4
```

<p align="right">(<a href="#top">back to top</a>)</p>

## Test

After training, you can also run the command below to test your model.

```shell
# distributed version
bash benchmarks/mmsegmentation/mim_dist_test.sh ${CONFIG} ${CHECKPOINT} ${GPUS}

# slurm version
bash benchmarks/mmsegmentation/mim_slurm_test.sh ${PARTITION} ${CONFIG} ${CHECKPOINT}
```

Remarks:

- `${CHECKPOINT}`: The trained segmentation model that you want to test. 

Example:

```shell
bash ./tools/benchmarks/mmsegmentation/mim_dist_test.sh \
configs/benchmarks/mmsegmentation/voc12aug/fcn_r50-d8_4xb4-20k_voc12aug-512x512.py \
work_dir/segmentation_model.pth 4
```

<p align="right">(<a href="#top">back to top</a>)</p>
