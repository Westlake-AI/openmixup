# Detection

- [Detection](#detection)
  - [MMDetection](#mmdetection)
    - [Evaluation](#evaluation)
  - [Detectron2](#detectron2)
    - [Evaluation](#evaluation)

## MMDetection

Here, we prefer to use MMDetection to do the detection task. First, make sure you have installed [MIM](https://github.com/open-mmlab/mim), which is also a project of OpenMMLab.

```shell
pip install openmim
mim install mmdet
```

It is very easy to install the package.

Besides, please refer to MMDet for [installation](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/get_started.md) and [data preparation](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/1_exist_data_model.md)

### Evaluation

After installing MMDet, you can run MMDetection with simple command. We provide scripts for the stage-4 only (`C4`) and `FPN` setting of object detection models.

```shell
# distributed version
bash benchmarks/mmdetection/mim_dist_train_c4.sh ${CONFIG} ${PRETRAIN} ${GPUS}
bash benchmarks/mmdetection/mim_dist_train.sh ${CONFIG} ${PRETRAIN} ${GPUS}

# slurm version
bash benchmarks/mmdetection/mim_slurm_train_c4.sh ${PARTITION} ${CONFIG} ${PRETRAIN}
bash benchmarks/mmdetection/mim_slurm_train.sh ${PARTITION} ${CONFIG} ${PRETRAIN}
```

Remarks:

- `CONFIG`: Use config files under `configs/benchmarks/mmdetection/` or write your own config files
- `PRETRAIN`: the pre-trained model file (the backbone parameters only).
- `${GPUS}`: The number of GPUs that you want to use to train. We adopt 8 GPUs for detection tasks by default.
- Since repositories of OpenMMLab have support referring config files across different repositories, we can easily leverage the configs from MMDetection like:
```shell
_base_ = 'mmdet::mask_rcnn/mask-rcnn_r50-caffe-c4_1x_coco.py'
```

Example:

```shell
bash ./tools/benchmarks/mmdetection/mim_dist_train_c4.sh \
configs/benchmarks/mmdetection/coco/mask-rcnn_r50-c4_ms-1x_coco.py \
https://download.openmmlab.com/mmselfsup/1.x/byol/byol_resnet50_16xb256-coslr-200e_in1k/byol_resnet50_16xb256-coslr-200e_in1k_20220825-de817331.pth 8
```

## Detectron2

If you want to do detection task with [detectron2](https://github.com/facebookresearch/detectron2), we also provide some config files.
Please refer to [INSTALL.md](https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md) for installation and follow the [directory structure](https://github.com/facebookresearch/detectron2/tree/main/datasets) to prepare your datasets required by detectron2.

```shell
conda activate detectron2 # use detectron2 environment here, otherwise use open-mmlab environment
cd tools/benchmarks/detectron2
python convert-pretrain-to-detectron2.py ${WEIGHT_FILE} ${OUTPUT_FILE} # must use .pkl as the output extension.
bash run.sh ${DET_CFG} ${OUTPUT_FILE}
```

<p align="right">(<a href="#top">back to top</a>)</p>

### Evaluation

After training, you can also run the command below with 8 GPUs (per node) to test your model.

```shell
# distributed version
bash benchmarks/mmdetection/mim_dist_test.sh ${CONFIG} ${CHECKPOINT}

# slurm version
bash benchmarks/mmdetection/slurm_run.sh ${PARTITION} ${CONFIG} ${CHECKPOINT}
```

Remarks:

- `${CHECKPOINT}`: The well-trained detection model that you want to test.

<p align="right">(<a href="#top">back to top</a>)</p>
