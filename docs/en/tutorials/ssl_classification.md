# SSL Downstream Tasks: Classification

The benchmarks of downstream tasks are based on [MMSelfSup](https://github.com/open-mmlab/mmselfsup). We provide many benchmarks to evaluate models on following downstream tasks. Here are comprehensive tutorials and examples to explain how to run all benchmarks with OpenMixup.

- [SSL Downstream Tasks: Classification](#ssl-downstream-tasks:-classification)
  - [ImageNet Linear Evaluation](#imagenet-linear-evaluation)
  - [ImageNet Finetune Evaluation](#imagenet-finetune-evaluation)
  - [ImageNet Semi-Supervised Classification](#imagenet-semi-supervised-classification)
  - [ImageNet Nearest-Neighbor Classification](#imagenet-nearest-neighbor-classification)
  - [VOC SVM / Low-shot SVM](#voc-svm--low-shot-svm)

First, you are supposed to extract your backbone weights by `tools/model_converters/extract_backbone_weights.py`

```shell
python tools/model_converters/extract_backbone_weights.py {CHECKPOINT} {MODEL_FILE}
```

Arguments:

- `CHECKPOINT`: the checkpoint file of a selfsup method named as epoch\_\*.pth.
- `MODEL_FILE`: the output backbone weights file. If not mentioned, the `PRETRAIN` below uses this extracted model file.

## ImageNet Linear Evaluation

The linear evaluation is one of the most general benchmarks for contrastive learning pre-training, we integrate several papers' config settings, also including multi-head linear evaluation. We write classification model in our own codebase for the multi-head function, thus, to run linear evaluation, we still use `.sh` script to launch training. The supported datasets are **ImageNet**, **Places205**, **iNaturalist18**, and **CIFAR-10/100**.

```shell
# distributed version
bash benchmarks/classification/dist_train_linear.sh ${CONFIG} ${PRETRAIN}

# slurm version
bash benchmarks/classification/srun_train_linear.sh ${PARTITION} ${JOB_NAME} ${CONFIG} ${PRETRAIN}
```

Remarks:

- The default GPU number is 8. When changing GPUS, please also change `imgs_per_gpu` in the config file accordingly to ensure the total batch size (e.g., 256).
- `CONFIG`: Use config files under `configs/benchmarks/classification/`. Specifically, `imagenet` (excluding `imagenet_*percent` folders), `places205`, `inaturalist2018`, and `CIFAR-10/100` are supported.
- `PRETRAIN`: the pre-trained model file (the backbone parameters only).

Example:

```shell
bash benchmarks/classification/dist_train_linear.sh \
configs/benchmarks/classification/imagenet/r50_linear_sz224_4xb64_step_ep100.py \
work_dir/pretrained_model.pth
```

<p align="right">(<a href="#top">back to top</a>)</p>

## ImageNet Finetune Evaluation

The fully finetuning evaluation is the popular benchmark for masked image modeling pre-training, we integrate several papers' config settings and use `.sh` script to launch training. The supported datasets are **ImageNet** and **CIFAR-10/100**.

```shell
# distributed version
bash benchmarks/classification/dist_train_ft_8gpu.sh ${CONFIG} ${PRETRAIN}

# slurm version
bash benchmarks/classification/srun_train_ft.sh ${PARTITION} ${JOB_NAME} ${CONFIG} ${PRETRAIN}
```

Remarks:

- The default GPU number is 8. When changing GPUS, please also change `imgs_per_gpu` in the config file accordingly to ensure the total batch size.
- `CONFIG`: Use config files under `configs/benchmarks/classification/`. Specifically, `imagenet` .
- `PRETRAIN`: the pre-trained model file.

Example:

```shell
bash benchmarks/classification/dist_train_ft_4gpu.sh \
configs/benchmarks/classification/imagenet/r50_rsb_a3_ft_sz160_4xb512_cos_fp16_ep100.py \
work_dir/pretrained_model.pth
```

<p align="right">(<a href="#top">back to top</a>)</p>

## ImageNet Semi-Supervised Classification

To run ImageNet semi-supervised classification, we still use `.sh` script as Linear Evaluation and Fine-tuning to launch training.

```shell
# distributed version
bash tools/benchmarks/classification/dist_train_semi.sh ${CONFIG} ${PRETRAIN}

# slurm version
bash tools/benchmarks/classification/slurm_train_semi.sh ${PARTITION} ${JOB_NAME} ${CONFIG} ${PRETRAIN}
```

Remarks:

- The default GPU number is 4.
- `CONFIG`: Use config files under `configs/benchmarks/classification/imagenet/`, named `imagenet_*percent` folders.
- `PRETRAIN`: the pre-trained model file.

<p align="right">(<a href="#top">back to top</a>)</p>

## ImageNet Nearest-Neighbor Classification

Only support CNN-style backbones (like ResNet50). To evaluate the pre-trained models using the nearest-neighbor benchmark, you can run command below.

```shell
# distributed version
bash benchmarks/classification/knn_imagenet/dist_test_knn_pretrain.sh ${SELFSUP_CONFIG} ${PRETRAIN}

# slurm version
bash tools/benchmarks/classification/knn_imagenet/slurm_test_knn_pretrain.sh ${PARTITION} ${JOB_NAME} ${SELFSUP_CONFIG} ${PRETRAIN}
```

Besides, if you want to evaluate the ckpt files saved by runner, you can run command below.

```shell
# distributed version
bash tools/benchmarks/classification/knn_imagenet/dist_test_knn_epoch.sh ${SELFSUP_CONFIG} ${EPOCH}

# slurm version
bash tools/benchmarks/classification/knn_imagenet/slurm_test_knn_epoch.sh ${PARTITION} ${JOB_NAME} ${SELFSUP_CONFIG} ${EPOCH}
```

**To test with ckpt, the code uses the epoch\_\*.pth file, there is no need to extract weights.**

Remarks:

- `${SELFSUP_CONFIG}` is the config file of the self-supervised experiment.
- `PRETRAIN`: the pre-trained model file.
- if you want to change GPU numbers, you could add `GPUS_PER_NODE=4 GPUS=4` at the beginning of the command.
- `EPOCH` is the epoch number of the ckpt that you want to test

<p align="right">(<a href="#top">back to top</a>)</p>

## VOC SVM / Low-shot SVM

To run these benchmarks, you should first prepare your VOC datasets. Please refer to [prepare_data.md](https://github.com/open-mmlab/mmselfsup/blob/master/docs/en/prepare_data.md) for the details of data preparation.

To evaluate the pre-trained models, you can run command below.

```shell
# distributed version
bash tools/benchmarks/classification/svm_voc07/dist_test_svm_pretrain.sh ${SELFSUP_CONFIG} ${GPUS} ${PRETRAIN} ${FEATURE_LIST}

# slurm version
bash tools/benchmarks/classification/svm_voc07/slurm_test_svm_pretrain.sh ${PARTITION} ${JOB_NAME} ${SELFSUP_CONFIG} ${PRETRAIN} ${FEATURE_LIST}
```

Besides, if you want to evaluate the ckpt files saved by runner, you can run command below.

```shell
# distributed version
bash tools/benchmarks/classification/svm_voc07/dist_test_svm_epoch.sh ${SELFSUP_CONFIG} ${EPOCH} ${FEATURE_LIST}

# slurm version
bash tools/benchmarks/classification/svm_voc07/slurm_test_svm_epoch.sh ${PARTITION} ${JOB_NAME} ${SELFSUP_CONFIG} ${EPOCH} ${FEATURE_LIST}
```

**To test with ckpt, the code uses the epoch\_\*.pth file, there is no need to extract weights.**

Remarks:

- `${SELFSUP_CONFIG}` is the config file of the self-supervised experiment.
- `${FEATURE_LIST}` is a string to specify features from layer1 to layer5 to evaluate; e.g., if you want to evaluate layer5 only, then `FEATURE_LIST` is "feat5", if you want to evaluate all features, then `FEATURE_LIST` is "feat1 feat2 feat3 feat4 feat5" (separated by space). If left empty, the default `FEATURE_LIST` is "feat5".
- `PRETRAIN`: the pre-trained model file.
- if you want to change GPU numbers, you could add `GPUS_PER_NODE=4 GPUS=4` at the beginning of the command.
- `EPOCH` is the epoch number of the ckpt that you want to test

<p align="right">(<a href="#top">back to top</a>)</p>
