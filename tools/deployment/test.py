import argparse
import warnings

import mmcv
import numpy as np
import torch
from mmcv import DictAction
from mmcv.parallel import MMDataParallel

from openmixup.apis.test import single_gpu_test_show
from openmixup.core.export import ONNXRuntimeClassifier, TensorRTClassifier
from openmixup.datasets import build_dataloader, build_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description='Test (and eval) an ONNX model using ONNXRuntime.')
    parser.add_argument('config', help='model config file')
    parser.add_argument('model', help='filename of the input ONNX model')
    parser.add_argument(
        '--backend',
        help='Backend of the model.',
        choices=['onnxruntime', 'tensorrt'])
    parser.add_argument(
        '--out', type=str, help='output result file in pickle format')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')
    parser.add_argument(
        '--metrics',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., '
        '"accuracy", "precision", "recall", "f1_score", "support" for single '
        'label dataset, and "mAP", "CP", "CR", "CF1", "OP", "OR", "OF1" for '
        'multi-label dataset')
    parser.add_argument(
        '--metric-options',
        nargs='+',
        action=DictAction,
        default={},
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be parsed as a dict metric_options for dataset.evaluate()'
        ' function.')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = mmcv.Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # build dataset and dataloader
    dataset = build_dataset(cfg.data.val)
    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=cfg.data.imgs_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

    # build onnxruntime model and run inference.
    if args.backend == 'onnxruntime':
        model = ONNXRuntimeClassifier(
            args.model, class_names=dataset.CLASSES, device_id=0)
    elif args.backend == 'tensorrt':
        model = TensorRTClassifier(
            args.model, class_names=dataset.CLASSES, device_id=0)
    else:
        print('Unknown backend: {}.'.format(args.model))
        exit(1)

    model = MMDataParallel(model, device_ids=[0])
    # model.CLASSES = dataset.CLASSES
    outputs = single_gpu_test_show(model, data_loader, args.show, args.show_dir)

    if args.metrics:
        results = dataset.evaluate(
            torch.from_numpy(np.vstack(outputs)),
            keyword='pred',
            logger=None,
            metric=args.metrics, metric_options=args.metric_options)
        for k, v in results.items():
            print(f'\n{k} : {v:.2f}')
    else:
        warnings.warn('Evaluation metrics are not specified.')
        scores = np.vstack(outputs)
        pred_score = np.max(scores, axis=1)
        pred_label = np.argmax(scores, axis=1)
        if dataset.CLASSES is not None:
            pred_class = [dataset.CLASSES[lb] for lb in pred_label]
        else:
            pred_class = pred_label
        results = {
            'pred_score': pred_score,
            'pred_label': pred_label,
            'pred_class': pred_class,
        }
        if not args.out:
            print('\nthe predicted result for the first element is '
                  f'pred_score = {pred_score[0]:.2f}, '
                  f'pred_label = {pred_label[0]} '
                  f'and pred_class = {pred_class[0]}. '
                  'Specify --out to save all results to files.')
    if args.out:
        print(f'\nwriting results to {args.out}')
        mmcv.dump(results, args.out)


if __name__ == '__main__':
    main()

# Usage: docs/en/tools/pytorch2onnx.md
