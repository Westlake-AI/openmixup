# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import mmcv
import numpy as np
import onnxruntime as rt
import torch
from mmcv.onnx import register_extra_symbolics
from mmcv.runner import load_checkpoint

from openmixup.models import build_model

torch.manual_seed(0)


def _demo_inputs(input_shape, num_classes):
    """Create a superset of inputs needed to run test or train batches.

    Args:
        input_shape (tuple): input batch dimensions
        num_classes (int): number of semantic classes
    """
    (N, C, H, W) = input_shape
    rng = np.random.RandomState(0)
    imgs = rng.rand(*input_shape)
    gt_label = rng.randint(
        low=0, high=num_classes, size=(N, 1)).astype(np.uint8)
    inputs = {
        'img': torch.FloatTensor(imgs).requires_grad_(True),
        'gt_label': torch.LongTensor(gt_label),
    }
    return inputs


def pytorch2onnx(model,
                 input_shape,
                 opset_version=11,
                 dynamic_export=False,
                 show=False,
                 output_file='tmp.onnx',
                 do_simplify=False,
                 verify=False):
    """Export Pytorch model to ONNX model and verify the outputs are same
    between Pytorch and ONNX.

    Args:
        model (nn.Module): Pytorch model we want to export.
        input_shape (tuple): Use this input shape to construct
            the corresponding dummy input and execute the model.
        opset_version (int): The onnx op version. Default: 11.
        show (bool): Whether print the computation graph. Default: False.
        output_file (string): The path to where we store the output ONNX model.
            Default: `tmp.onnx`.
        verify (bool): Whether compare the outputs between Pytorch and ONNX.
            Default: False.
    """
    model.cpu().eval()

    if hasattr(model.head, 'num_classes'):
        num_classes = model.head.num_classes
    # Some backbones use `num_classes=-1` to disable top classifier.
    elif getattr(model.backbone, 'num_classes', -1) > 0:
        num_classes = model.backbone.num_classes
    else:
        raise AttributeError('Cannot find "num_classes" in both head and '
                             'backbone, please check the config file.')

    inputs = _demo_inputs(input_shape, num_classes)

    img = inputs.pop('img')

    # replace original forward function
    origin_forward = model.forward
    model.forward = model.forward_inference
    register_extra_symbolics(opset_version)

    # support dynamic shape export
    if dynamic_export:
        dynamic_axes = {
            'input': {
                0: 'batch',
                2: 'width',
                3: 'height'
            },
            'probs': {
                0: 'batch'
            }
        }
    else:
        dynamic_axes = {}

    with torch.no_grad():
        torch.onnx.export(
            model, (img, ),
            output_file,
            input_names=['input'],
            output_names=['probs'],
            export_params=True,
            keep_initializers_as_inputs=True,
            dynamic_axes=dynamic_axes,
            verbose=show,
            opset_version=opset_version)
        print(f'Successfully exported ONNX model: {output_file}')
    model.forward = origin_forward

    if do_simplify:
        import onnx
        import onnxsim
        from mmcv import digit_version

        min_required_version = '0.4.0'
        assert digit_version(onnxsim.__version__) >= digit_version(
            min_required_version
        ), f'Requires to install onnxsim>={min_required_version}'

        model_opt, check_ok = onnxsim.simplify(output_file)
        if check_ok:
            onnx.save(model_opt, output_file)
            print(f'Successfully simplified ONNX model: {output_file}')
        else:
            print('Failed to simplify ONNX model.')
    if verify:
        # check by onnx
        import onnx
        onnx_model = onnx.load(output_file)
        onnx.checker.check_model(onnx_model)

        # test the dynamic model
        if dynamic_export:
            dynamic_test_inputs = _demo_inputs(
                (input_shape[0], input_shape[1], input_shape[2] * 2,
                 input_shape[3] * 2), model.head.num_classes)
            img = dynamic_test_inputs.pop('img')

        # check the numerical value
        # get pytorch output
        pytorch_result = model(img, mode='inference').detach().numpy()

        # get onnx output
        input_all = [node.name for node in onnx_model.graph.input]
        input_initializer = [
            node.name for node in onnx_model.graph.initializer
        ]
        net_feed_input = list(set(input_all) - set(input_initializer))
        assert (len(net_feed_input) == 1)
        sess = rt.InferenceSession(output_file)
        onnx_result = sess.run(
            None, {net_feed_input[0]: img.detach().numpy()})[0]
        if not np.allclose(pytorch_result, onnx_result):
            raise ValueError(
                'Verification: The outputs are different between Pytorch and ONNX')
        print('Verification: The outputs are same between Pytorch and ONNX.')


def parse_args():
    parser = argparse.ArgumentParser(description='Convert MMCls to ONNX')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file', default=None)
    parser.add_argument('--show', action='store_true', help='show onnx graph')
    parser.add_argument(
        '--verify', action='store_true', help='verify the onnx model')
    parser.add_argument('--output-file', type=str, default='tmp.onnx')
    parser.add_argument('--opset-version', type=int, default=11)
    parser.add_argument(
        '--simplify',
        action='store_true',
        help='Whether to simplify onnx model.')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[224, 224],
        help='input image size')
    parser.add_argument(
        '--dynamic-export',
        action='store_true',
        help='Whether to export ONNX with dynamic input shape. \
            Defaults to False.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    if len(args.shape) == 1:
        input_shape = (1, 3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (
            1,
            3,
        ) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    cfg = mmcv.Config.fromfile(args.config)
    cfg.model.pretrained = None

    # build the model and load checkpoint
    classifier = build_model(cfg.model)

    if args.checkpoint:
        load_checkpoint(classifier, args.checkpoint, map_location='cpu')

    # convert model to onnx file
    pytorch2onnx(
        classifier,
        input_shape,
        opset_version=args.opset_version,
        show=args.show,
        dynamic_export=args.dynamic_export,
        output_file=args.output_file,
        do_simplify=args.simplify,
        verify=args.verify)

# Usage: docs/en/tools/pytorch2onnx.md
