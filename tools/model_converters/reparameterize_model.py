import argparse
import torch
from mmcv import Config
from mmcv.runner import load_checkpoint
from pathlib import Path

from openmixup.models import build_model
from openmixup.models.classifiers import Classification


def convert_classifier_to_deploy(model, save_path):
    print('Converting...')
    assert hasattr(model, 'backbone') and \
        hasattr(model.backbone, 'switch_to_deploy'), \
        '`model.backbone` must has method of "switch_to_deploy".' \
        f' But {model.backbone.__class__} does not have.'

    model.backbone.switch_to_deploy()
    torch.save(model.state_dict(), save_path)

    print('Done! Save at path "{}"'.format(save_path))


def main():
    parser = argparse.ArgumentParser(
        description='Convert the parameters of the RepVGG or RepMLP block '
        'from training mode to deployment mode.')
    parser.add_argument(
        'config_path',
        help='The path to the configuration file of the network '
        'containing the repvgg block.')
    parser.add_argument(
        'checkpoint_path',
        help='The path to the checkpoint file corresponding to the model.')
    parser.add_argument(
        'save_path',
        help='The path where the converted checkpoint file is stored.')
    args = parser.parse_args()

    save_path = Path(args.save_path)
    if save_path.suffix != '.pth':
        print('The path should contain the name of the pth format file.')
        exit()
    save_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = Config.fromfile(args.config)
    model = build_model(cfg.model)
    if args.load_checkpoint is not None:
        assert isinstance(args.load_checkpoint, str) and args.pretrained is None
        load_checkpoint(model, args.load_checkpoint, map_location='cpu')
    assert isinstance(model, Classification), \
        '`model` must be a `openmixup.classifiers.Classification` instance.'

    convert_classifier_to_deploy(model, args.save_path)


if __name__ == '__main__':
    main()
