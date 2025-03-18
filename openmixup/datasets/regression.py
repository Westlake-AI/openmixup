import torch
from openmixup.models.utils import (pearson_correlation, \
                                    spearman_correlation, regression_error)
from openmixup.utils import print_log

from .registry import DATASETS
from .base import BaseDataset
from .utils import to_numpy, to_tensor


@DATASETS.register_module
class RegressionDataset(BaseDataset):
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
        super(RegressionDataset, self).__init__(data_source, pipeline, prefetch)

    def __getitem__(self, idx):
<<<<<<< HEAD
        data, target = self.data_source.get_sample(idx)
        data = self.pipeline(data)
        if self.prefetch:
            data = torch.from_numpy(to_numpy(data))
        return dict(data=data, gt_label=target, idx=idx)
=======
        img, target = self.data_source.get_sample(idx)
        img = self.pipeline(img)
        if self.prefetch:
            img = torch.from_numpy(to_numpy(img))
        return dict(img=img, gt_label=target, idx=idx)
>>>>>>> db2c4ac (update some vit-based mixup methods and fix robustness eval tasks)

    def evaluate(self,
                 scores, keyword, logger=None,
                 metric='mse',
                 metric_options=None,
                 indices=None,
                 **kwargs):
        """The evaluation function to output regression error.

        Args:
            scores (tensor): Prediction values.
            keyword (str): The corresponding head name.
            logger (logging.Logger | str | None, optional): The defined logger
                to be used. Defaults to None.
            metric (str | list[str]): Metrics to be evaluated.
                Default value is `mse`.
            metric_options (dict, optional): Options for calculating metrics.
                The allowed key is 'average_mode'. Defaults to None.
        Returns:
            dict: evaluation results
        """
        if metric_options is None:
            metric_options = dict(average_mode='mean')
        if isinstance(metric, str):
            metrics = [metric]
        else:
            metrics = metric
        eval_res = {}
        eval_log = []
<<<<<<< HEAD
        allowed_metrics = ['mse', 'mae', 'pearson', 'spearman',]
=======
        allowed_metrics = ['mse', 'mae', 'rmse', 'mape', 'pearson', 'spearman',]
>>>>>>> db2c4ac (update some vit-based mixup methods and fix robustness eval tasks)
        average_mode = metric_options.get('average_mode', 'mean')
        invalid_metrics = set(metrics) - set(allowed_metrics)
        if len(invalid_metrics) != 0:
            raise ValueError(f'metric {invalid_metrics} is not supported.')
        
        target = to_tensor(self.data_source.labels).type(torch.float32).squeeze()
        if indices is not None:
            target = target[indices]
        assert scores.size(0) == target.size(0), \
            "Inconsistent length for results and labels, {} vs {}".format(
            scores.size(0), target.size(0))
        
<<<<<<< HEAD
        mse, mae = regression_error(scores, target, average_mode=average_mode)
=======
        mse, mae, rmse, mape = regression_error(scores, target, average_mode=average_mode)
>>>>>>> db2c4ac (update some vit-based mixup methods and fix robustness eval tasks)
        if 'mse' in metrics:
            eval_res[f"{keyword}_mse"] = float(mse)
            eval_log.append("{}_mse: {:.03f}".format(keyword, float(mse)))
        if 'mae' in metrics:
            eval_res[f"{keyword}_mae"] = float(mae)
            eval_log.append("{}_mae: {:.03f}".format(keyword, float(mae)))
<<<<<<< HEAD
=======
        if 'rmse' in metrics:
            eval_res[f"{keyword}_rmse"] = float(rmse)
            eval_log.append("{}_rmse: {:.03f}".format(keyword, float(rmse)))
        if 'mape' in metrics:
            eval_res[f"{keyword}_mape"] = float(mape)
            eval_log.append("{}_mape: {:.03f}".format(keyword, float(mape)))
>>>>>>> db2c4ac (update some vit-based mixup methods and fix robustness eval tasks)
        if 'pearson' in metrics:
            p_corr = pearson_correlation(scores, target, average_mode=average_mode)
            eval_res[f"{keyword}_pearson"] = float(p_corr)
            eval_log.append("{}_pearson: {:.03f}".format(keyword, float(p_corr)))
        if 'spearman' in metrics:
            s_corr = spearman_correlation(scores, target, average_mode=average_mode)
            eval_res[f"{keyword}_spearman"] = float(s_corr)
            eval_log.append("{}_spearman: {:.03f}".format(keyword, float(s_corr)))
        
        if logger is not None and logger != 'silent':
            for _log in eval_log:
                print_log(_log, logger=logger)
        
        return eval_res
