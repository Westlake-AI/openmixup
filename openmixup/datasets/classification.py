import numpy as np
import torch
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.metrics as metric

from openmixup.models.utils import precision_recall_f1, support
from openmixup.utils import print_log

from .registry import DATASETS
from .base import BaseDataset
from .utils import to_numpy


@DATASETS.register_module
class ClassificationDataset(BaseDataset):
    """The dataset outputs one view of an image, containing some other
        information such as label, idx, etc.

    Args:
        data_source (dict): Data source defined in
            `mmselfsup.datasets.data_sources`.
        pipeline (list[dict]): A list of dict, where each element represents
            an operation defined in `mmselfsup.datasets.pipelines`.
        prefetch (bool, optional): Whether to prefetch data. Defaults to False.
    """

    def __init__(self, data_source, pipeline, prefetch=False):
        super(ClassificationDataset, self).__init__(data_source, pipeline, prefetch)

    def __getitem__(self, idx):
        img, target = self.data_source.get_sample(idx)
        assert isinstance(img, Image.Image), \
            'The output from the data source must be an Image, got: {}. \
            Please ensure that the list file does not contain labels.'.format(
            type(img))
        img = self.pipeline(img)
        if self.prefetch:
            img = torch.from_numpy(to_numpy(img))
        return dict(img=img, gt_label=target, idx=idx)

    def evaluate(self,
                 scores, keyword, logger=None,
                 metric='accuracy', metric_options=None, topk=(1, 5),
                 **kwargs):
        """The evaluation function to output accuracy.

        Args:
            scores (tensor): The prediction values of output heads in (N, \*).
            keyword (str): The corresponding head name in (N, \*).
            logger (logging.Logger | str | None, optional): The defined logger
                to be used. Defaults to None.
            metric (str | list[str]): Metrics to be evaluated. Default to `accuracy`.
            metric_options (dict, optional): Options for calculating metrics.
                Allowed keys are 'thrs' and 'average_mode'. Defaults to None.
            topk (tuple(int)): The output includes topk accuracy.
        Returns:
            dict: evaluation results
        """
        if metric_options is None:
            metric_options = dict(average_mode='macro')
        if isinstance(metric, str):
            metrics = [metric]
        else:
            metrics = metric
        eval_res = {}
        eval_log = []
        allowed_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'support',]
        average_mode = metric_options.get('average_mode', 'macro')
        invalid_metrics = set(metrics) - set(allowed_metrics)
        if len(invalid_metrics) != 0:
            raise ValueError(f'metric {invalid_metrics} is not supported.')
        
        target = torch.LongTensor(self.data_source.labels)
        assert scores.size(0) == target.size(0), \
            "Inconsistent length for results and labels, {} vs {}".format(
            scores.size(0), target.size(0))
        
        if 'accuracy' in metrics:
            _, pred = scores.topk(max(topk), dim=1, largest=True, sorted=True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))  # KxN
            for k in topk:
                correct_k = correct[:k].contiguous().view(-1).float().sum(0).item()
                acc = correct_k * 100.0 / scores.size(0)
                eval_res[f"{keyword}_top{k}"] = acc
                eval_log.append("{}_top{}: {:.03f}".format(keyword, k, acc))
        
        if 'support' in metrics:
            support_value = support(scores, target, average_mode=average_mode)
            eval_res[f'{keyword}_support'] = support_value
            eval_log.append("{}_support: {:.03f}".format(keyword, support_value))
        
        precision_recall_f1_keys = ['precision', 'recall', 'f1_score']
        if len(set(metrics) & set(precision_recall_f1_keys)) != 0:
            thrs = metric_options.get('thrs', 0.)
            if thrs is not None:
                precision_recall_f1_values = precision_recall_f1(
                    scores, target, average_mode=average_mode, thrs=thrs)
            else:
                precision_recall_f1_values = precision_recall_f1(
                    scores, target, average_mode=average_mode)
            for key, values in zip(precision_recall_f1_keys,
                                   precision_recall_f1_values):
                if key in metrics:
                    if isinstance(thrs, tuple):
                        eval_res.update({f'{key}_thr_{thr:.2f}': value
                            for thr, value in zip(thrs, values)
                        })
                    else:
                        eval_res[key] = values
                        eval_log.append("{}_{}: {:.03f}".format(keyword, key, values))
        
        if logger is not None and logger != 'silent':
            for _log in eval_log:
                print_log(_log, logger=logger)
        
        return eval_res

    def ece_score(self, py, n_bins=10, save_name='.', show_plot=True):
        py = torch.tensor(py).cuda().cpu()
        py = py.softmax(dim=1)
        y_test = torch.LongTensor(self.data_source.labels)

        py = np.array(py)
        y_test = np.array(y_test)
        if y_test.ndim > 1:
            y_test = np.argmax(y_test, axis=1)
        py_index = np.argmax(py, axis=1)
        py_value = []

        for i in range(py.shape[0]):
            py_value.append(py[i, py_index[i]])
        py_value = np.array(py_value)
        acc, conf = np.zeros(n_bins), np.zeros(n_bins)
        Bm = np.zeros(n_bins)
        x = []
        for m in range(n_bins):
            a, b = m / n_bins, (m + 1) / n_bins
            x.append(a)
            for i in range(py.shape[0]):
                if py_value[i] > a and py_value[i] <= b:
                    Bm[m] += 1
                    if py_index[i] == y_test[i]:
                        acc[m] += 1
                    conf[m] += py_value[i]

            if Bm[m] != 0:
                acc[m] = acc[m] / Bm[m]
                conf[m] = conf[m] / Bm[m]
        acc.sort()
        conf.sort()
        ece = 0
        for m in range(n_bins):
            ece += Bm[m] * np.abs((acc[m] - conf[m]))

        x.append(1.0)
        if show_plot:
            plt.figure(figsize=(5, 5))
            sns.set_style("whitegrid", rc={'grid.linestyle': '--',
                                    "axes.edgecolor": '.20',})
            plt.plot(x, x, color='r', linestyle='--', linewidth=1)
            plt.plot(acc, conf, color='b', linestyle='-', linewidth=1)
            plt.savefig(f"{save_name}/ece_score.png", format='png', dpi=300)
            plt.show()

        return ece / sum(Bm)
