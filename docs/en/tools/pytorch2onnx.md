# Pytorch to ONNX (Experimental)

- [Pytorch to ONNX (Experimental)](#pytorch-to-onnx-experimental)
  - [How to convert models from Pytorch to ONNX](#how-to-convert-models-from-pytorch-to-onnx)
    - [Prerequisite](#prerequisite)
    - [Usage](#usage)
    - [Description of all arguments:](#description-of-all-arguments)
  - [How to evaluate ONNX models with ONNX Runtime](#how-to-evaluate-onnx-models-with-onnx-runtime)
    - [Prerequisite](#prerequisite-1)
    - [Usage](#usage-1)
    - [Description of all arguments](#description-of-all-arguments-1)
  - [Reminders](#reminders)
  - [FAQs](#faqs)


## How to convert models from Pytorch to ONNX

### Prerequisite

1. Please refer to [install](https://mmclassification.readthedocs.io/en/latest/install.html#install-mmclassification) for installation of MMClassification.
2. Install onnx, onnxsim (optional for `--simplify`), and onnxruntime.

```shell
pip install onnx onnxsim onnxruntime==1.5.1
```

### Usage

```bash
python tools/deployment/pytorch2onnx.py \
    ${CONFIG_FILE} \
    --checkpoint ${CHECKPOINT_FILE} \
    --output-file ${OUTPUT_FILE} \
    --shape ${IMAGE_SHAPE} \
    --opset-version ${OPSET_VERSION} \
    --dynamic-export \
    --simplify \
    --verify \
```

### Description of all arguments:

- `config` : The path of a model config file.
- `--checkpoint` : The path of a model checkpoint file.
- `--output-file`: The path of output ONNX model. If not specified, it will be set to `tmp.onnx`.
- `--shape`: The height and width of input tensor to the model. If not specified, it will be set to `224 224`.
- `--opset-version` : The opset version of ONNX. If not specified, it will be set to `11`.
- `--dynamic-export` : Determines whether to export ONNX with dynamic input shape and output shapes. If not specified, it will be set to `False`.
- `--simplify`: Determines whether to simplify the exported ONNX model. If not specified, it will be set to `False`.
- `--verify`: Determines whether to verify the correctness of an exported model. If not specified, it will be set to `False`.

Example:

```bash
python tools/deployment/pytorch2onnx.py \
    configs/classification/imagenet/mixups/basic/r18_mixups_CE_none_4xb64.py \
    --checkpoint ${PATH_TO_MODEL}/r18_mixups_CE_none_4xb64.pth \
    --output-file ${PATH_TO_MODEL}/r18_mixups_CE_none_4xb64.onnx \
    --dynamic-export \
    --simplify \
    --verify \
```

## How to evaluate ONNX models with ONNX Runtime

We prepare a tool `tools/deployment/test.py` to evaluate ONNX models with ONNXRuntime or TensorRT.

### Prerequisite

- Install onnx and onnxruntime-gpu accordingt to [instructions](https://onnxruntime.ai/) for ONNXRuntime.

  ```shell
  pip install onnx onnxruntime-gpu
  ```
- Install tensorrt according to [PyTorch instructions](https://pytorch.org/TensorRT/getting_started/installation.html#installation) for TensorRT evaluations.

### Usage

```bash
python tools/deployment/test.py \
    ${CONFIG_FILE} \
    ${ONNX_FILE} \
    --backend ${BACKEND} \
    --out ${OUTPUT_FILE} \
    --metrics ${EVALUATION_METRICS} \
    --metric-options ${EVALUATION_OPTIONS} \
    --show
    --show-dir ${SHOW_DIRECTORY} \
    --cfg-options ${CFG_OPTIONS} \
```

### Description of all arguments

- `config_file`: The path of a model config file.
- `onnx_file`: The path of a ONNX model file.
- `--backend`: Backend for input model to run and should be `onnxruntime` or `tensorrt`.
- `--out`: The path of output result file in pickle format (e.g., `.pkl`).
- `--metrics`: Evaluation metrics, which depends on the dataset, e.g., "accuracy", "precision", "recall", "f1_score", "support" for single label dataset.
- `--metrics-options`: Custom options for evaluation, the key-value pair in `xxx=yyy` format will be kwargs for `dataset.evaluate()` function.
- `--show`: Determines whether to show classifier outputs. If not specified, it will be set to `False`.
- `--show-dir`: Directory where painted images will be saved.
- `--cfg-options`: Override some settings in the used config file, the key-value pair in `xxx=yyy` format will be merged into config file.

Example:

```bash
python tools/deployment/test.py \
    configs/classification/imagenet/mixups/basic/r18_mixups_CE_none_4xb64.py \
    ${PATH_TO_MODEL}/r18_mixups_CE_none_4xb64.onnx \
    --backend onnxruntime \
    --out ${PATH_TO_MODEL}/out.pkl \
    --show-dir ${SHOW_DIRECTORY} \
    --metrics accuracy
```

## Reminders

- If you meet any problem with the listed models above, please create an issue and it would be taken care of soon. For models not included in the list, please try to dig a little deeper and debug a little bit more and hopefully solve them by yourself.

## FAQs

- None
