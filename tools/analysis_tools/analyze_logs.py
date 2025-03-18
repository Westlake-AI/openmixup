"""
Analyze statistics from some log.json files

Example 1: Plot top-1 accuracy of `exp_1` and `exp_2`
python tools/analysis_tools/analyze_logs.py plot_curve exp_1.log.json exp_2.log.json --out tmp.png --key acc_top1


Example 1: Print training times of `exp_1`
python tools/analysis_tools/analyze_logs.py cal_train_time exp_1.log.json
"""
import argparse
import json
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def cal_train_time(log_dicts, args):
    """Compute the average time per training iteration."""
    for i, log_dict in enumerate(log_dicts):
        print(f'{"-" * 5}Analyze train time of {args.json_logs[i]}{"-" * 5}')
        all_times = []
        for epoch in log_dict.keys():
            if args.include_outliers:
                all_times.append(log_dict[epoch]['time'])
            else:
                all_times.append(log_dict[epoch]['time'][1:])
        all_times = np.array(all_times)
        epoch_ave_time = all_times.mean(-1)
        slowest_epoch = epoch_ave_time.argmax()
        fastest_epoch = epoch_ave_time.argmin()
        std_over_epoch = epoch_ave_time.std()
        print(f'slowest epoch {slowest_epoch + 1}, '
              f'average time is {epoch_ave_time[slowest_epoch]:.4f}')
        print(f'fastest epoch {fastest_epoch + 1}, '
              f'average time is {epoch_ave_time[fastest_epoch]:.4f}')
        print(f'time std over epochs is {std_over_epoch:.4f}')
        print(f'average iter time: {np.mean(all_times):.4f} s/iter')
        print()


def plot_curve(log_dicts, args):
    """Plot train metric-iter graph."""
    if args.backend is not None:
        plt.switch_backend(args.backend)
    sns.set_style(args.style)
    # if legend is None, use {filename}_{key} as legend
    legend = args.legend
    if legend is None:
        legend = []
        for json_log in args.json_logs:
            # for metric in args.keys:
            filename = (json_log.split('/')[-1]).split('.json')[0]
            legend.append(f'{filename}')
    # assert len(legend) == (len(args.json_logs) * len(args.keys))
    metrics = args.keys
    num_metrics = len(metrics)
    for i, log_dict in enumerate(log_dicts):
        epochs = list(log_dict.keys())
        for j, metric in enumerate(metrics):
            print(f'plot curve of {args.json_logs[i]}, metric is {metric}')
            if any(m in metric for m in ('mAP', 'epoch')):
                xs = epochs
                ys = [log_dict[e]['acc_top1'] for e in xs]
                ax = plt.gca()
                # ax.set_xticks([1,10,20,30,40,50,60,70,80,90,100])
                ax.set_xticks([1, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200])
                plt.xlabel('Epoch', size=20)
                plt.ylabel('Top-1 Accuracy(%)', size=20)
                plt.plot(xs, ys, label=legend[i * num_metrics + j], linewidth=1.5)
            elif metric == 'acc_top1':
                xs = []
                ys = []
                num_iters_per_epoch = log_dict[epochs[0]]['iter'][-1]
                for epoch in epochs:
                    iters = log_dict[epoch]['iter']
                    if log_dict[epoch]['mode'][-1] == 'val':
                        iters = iters[:-1]
                    assert len(iters) > 0, (
                        'The training log is empty, please try to reduce the '
                        'interval of log in config file.')
                    res = log_dict[epoch][metric][:len(iters)]
                    if len(iters) > len(res):
                        iters = iters[-1:]
                        res = res[-1:]
                    xs.append(
                        np.array(iters) + (epoch - 1) * num_iters_per_epoch)
                    ys.append(np.array(res))
                xs = np.concatenate(xs)
                ys = np.concatenate(ys)
                plt.xlabel('Iter', size=20)
                plt.ylabel('Top-1 Accuracy(%)', size=20)
                plt.plot(
                    xs, ys, label=legend[i * num_metrics + j], linewidth=1)
            elif metric == 'cos_simi_weight':
                xs = []
                ys = []
                num_iters_per_epoch = log_dict[epochs[0]]['iter'][-1]
                for epoch in epochs:
                    iters = log_dict[epoch]['iter']
                    if log_dict[epoch]['mode'][-1] == 'val':
                        iters = iters[:-1]
                    assert len(iters) > 0, (
                        'The training log is empty, please try to reduce the '
                        'interval of log in config file.')
                    res = log_dict[epoch][metric][:len(iters)]
                    if len(iters) > len(res):
                        iters = iters[-1:]
                        res = res[-1:]
                    xs.append(
                        np.array(iters) + (epoch - 1) * num_iters_per_epoch)
                    ys.append(np.array(res))
                xs = np.concatenate(xs)
                ys = np.concatenate(ys)
                plt.xlabel('Cos_simi_weight')
                plt.ylabel('Alpha')
                plt.plot(
                    xs, ys, label=legend[i * num_metrics + j], linewidth=1)
            else:
                xs = []
                ys = []
                num_iters_per_epoch = log_dict[epochs[0]]['iter'][-1]
                for epoch in epochs:
                    iters = log_dict[epoch]['iter']
                    if log_dict[epoch]['mode'][-1] == 'val':
                        iters = iters[:-1]
                    assert len(iters) > 0, (
                        'The training log is empty, please try to reduce the '
                        'interval of log in config file.')
                    res = log_dict[epoch][metric][:len(iters)]
                    if len(iters) > len(res):
                        iters = iters[-1:]
                        res = res[-1:]
                    xs.append(
                        np.array(iters) + (epoch - 1) * num_iters_per_epoch)
                    ys.append(np.array(res))
                xs = np.concatenate(xs)
                ys = np.concatenate(ys)
                plt.xlabel('Iter', size=15)
                plt.ylabel('Loss', size=15)
                plt.plot(
                    xs, ys, label=legend[i * num_metrics + j], linewidth=2)
            # 显示label
            plt.legend()
        if args.title is not None:
            plt.title(args.title)
    if args.out is None:
        plt.show()
    else:
        print(f'save curve to: {args.out}')
        plt.savefig(args.out)
        plt.cla()


def add_plot_parser(subparsers):
    parser_plt = subparsers.add_parser(
        'plot_curve', help='parser for plotting curves')
    parser_plt.add_argument(
        'json_logs',
        type=str,
        nargs='+',
        help='path of train log in json format')
    parser_plt.add_argument(
        '--keys',
        type=str,
        nargs='+',
        default=['acc_top1'],
        help='the metric that you want to plot')
    parser_plt.add_argument('--title', default='', type=str, nargs='+', help='title of figure')
    parser_plt.add_argument(
        '--legend',
        type=str,
        nargs='+',
        default=None,
        help='legend of each plot')
    parser_plt.add_argument(
        '--backend', type=str, default=None, help='backend of plt')
    parser_plt.add_argument(
        '--style', type=str, default='whitegrid', help='style of plt')
    parser_plt.add_argument('--out', type=str, default=None)


def add_time_parser(subparsers):
    parser_time = subparsers.add_parser(
        'cal_train_time',
        help='parser for computing the average time per training iteration')
    parser_time.add_argument(
        'json_logs',
        type=str,
        nargs='+',
        help='path of train log in json format')
    parser_time.add_argument(
        '--include-outliers',
        action='store_true',
        help='include the first value of every epoch when computing '
        'the average time')


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze Json Log')
    # currently only support plot curve and calculate average train time
    subparsers = parser.add_subparsers(dest='task', help='task parser')
    add_plot_parser(subparsers)
    add_time_parser(subparsers)
    args = parser.parse_args()
    return args


def load_json_logs(json_logs, args):
    """load and convert json_logs to log_dicts.

    Args:
        json_logs (str): paths of json_logs.

    Returns:
        list[dict(int:dict())]: key is epoch, value is a sub dict keys of
            sub dict is different metrics, e.g. memory, bbox_mAP, value of
            sub dict is a list of corresponding values of all iterations.
    """
    log_dicts = [dict() for _ in json_logs]
    for json_log, log_dict in zip(json_logs, log_dicts):
        with open(json_log, 'r') as log_file:
            for line in log_file:
                log = json.loads(line.strip())
                # skip lines without `epoch` field
                if 'epoch' not in log:
                    continue
                epoch = log.pop('epoch')
                if epoch not in log_dict:
                    log_dict[epoch] = defaultdict(list)
                for k, v in log.items():
                    # change a new name
                    if k == 'head0_top1' or k =='acc_one_k_top1':
                        k = 'acc_top1'
                        # k = args.keys[0]
                    log_dict[epoch][k].append(v)
    return log_dicts


def main():
    args = parse_args()

    json_logs = args.json_logs
    for json_log in json_logs:
        assert json_log.endswith('.json')

    log_dicts = load_json_logs(json_logs, args)

    eval(args.task)(log_dicts, args)


if __name__ == '__main__':
    main()
