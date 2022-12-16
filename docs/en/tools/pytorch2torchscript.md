# Pytorch to TorchScript (Experimental)

- [Pytorch to TorchScript (Experimental)](#pytorch-to-torchscript-experimental)
  - [How to convert models from Pytorch to TorchScript](#how-to-convert-models-from-pytorch-to-torchscript)
    - [Usage](#usage)
    - [Description of all arguments](#description-of-all-arguments)
  - [Reminders](#reminders)
  - [FAQs](#faqs)


## How to convert models from Pytorch to TorchScript

### Usage

```bash
python tools/deployment/pytorch2torchscript.py \
    ${CONFIG_FILE} \
    --checkpoint ${CHECKPOINT_FILE} \
    --output-file ${OUTPUT_FILE} \
    --shape ${IMAGE_SHAPE} \
    --verify \
```

### Description of all arguments

- `config` : The path of a model config file.
- `--checkpoint` : The path of a model checkpoint file.
- `--output-file`: The path of output TorchScript model. If not specified, it will be set to `tmp.pt`.
- `--shape`: The height and width of input tensor to the model. If not specified, it will be set to `224 224`.
- `--verify`: Determines whether to verify the correctness of an exported model. If not specified, it will be set to `False`.

Example:

```bash
python tools/deployment/pytorch2torchscript.py \
    configs/classification/imagenet/mixups/basic/r18_mixups_CE_none_4xb64.py \
    --checkpoint ${PATH_TO_MODEL}/r18_mixups_CE_none_4xb64.pth \
    --output-file ${PATH_TO_MODEL}/r18_mixups_CE_none_4xb64.pt \
    --verify \
```

Notes:

- We have tested the most models with Pytorch==1.10.0*.

## Reminders

- For torch.jit.is_tracing() is only supported after v1.6. For users with pytorch v1.3-v1.5, we suggest early returning tensors manually.
- If you meet any problem with the models in this repo, please create an issue and it would be taken care of soon.

## FAQs

- None
