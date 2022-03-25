import numpy as np
import mmcv
import torch
from mmcv.runner import Hook
from torch.utils.data import Dataset
from matplotlib import pyplot as plt

from openmixup.utils import nondist_forward_collect, dist_forward_collect
from openmixup import datasets
from openmixup.third_party import WeightedKNNClassifier
from .registry import HOOKS


@HOOKS.register_module
class SSLMetricHook(Hook):
    """Self-supervised learning metrics hook.

    Args:
        val_dataset (Dataset | dict): A PyTorch dataset or dict that indicates
            the dataset.
        train_dataset (Dataset | dict): A PyTorch dataset or dict that indicates
            the dataset.
        dist_mode (bool): Use distributed evaluation or not. Default: True.
        forward_mode (str): Mode of forward to extract features, e.g., SSL
            methods use `extract` mode for backbone features. Default: 'test'.
        metric_mode (str): Mode of linear classification. Default: 'knn'.
        metric_args (dict): Dict of arguments for metric tools. Default: None.
        visual_mode (str): Mode of embedding visualization. Default: 'umap'.
        visual_args (dict): Dict of arguments for visualization methods. Notice
            that the target dimension is 2. Default: None.
        initial (bool): Whether to evaluate before the training starts.
            Default: True.
        interval (int): Evaluation interval (by epochs). Default: 1.
        save_val (bool): Whether to save evaluation results. Default: False.
        **eval_kwargs: Evaluation arguments fed into the evaluate function of
            the dataset.
    """

    def __init__(self,
                 val_dataset,
                 train_dataset=None,
                 dist_mode=True,
                 forward_mode='test',
                 metric_mode="knn",
                 metric_args=dict(knn=20),
                 visual_mode="tsne",
                 visual_args=dict(n_components=2),
                 initial=True,
                 interval=1,
                 save_val=False,
                 **eval_kwargs):
        self.forward_mode = forward_mode
        self.metric_mode = metric_mode
        self.visual_mode = visual_mode
        assert forward_mode in ['test', 'vis', 'extract',]
        assert metric_mode is None or metric_mode in ["knn",]
        assert visual_mode is None or visual_mode in ["tsne", "umap",]
        self.dist_mode = dist_mode
        self.initial = initial
        self.interval = interval
        self.save_val = save_val
        self.eval_kwargs = eval_kwargs
        self.visual_args = visual_args
        # build datasets
        self.val_dataset, self.val_data_loader = \
            self._build_dataloader(val_dataset, eval_kwargs)
        if self.metric_mode is not None:
            self.train_dataset, self.train_data_loader = \
                self._build_dataloader(train_dataset, eval_kwargs)
        # build metrics
        self._build_metric_tools(metric_args)
        # build visualization
        self._build_visualization_tools(visual_args)

    def _build_dataloader(self, dataset, eval_kwargs):
        if isinstance(dataset, Dataset):
            pass
        elif isinstance(dataset, dict):
            dataset = datasets.build_dataset(dataset)
        else:
            raise TypeError(
                f'dataset must be a Dataset object or a dict, not {type(dataset)}')
        data_loader = datasets.build_dataloader(
            dataset,
            eval_kwargs['imgs_per_gpu'],
            eval_kwargs['workers_per_gpu'],
            dist=self.dist_mode,
            shuffle=False,
            prefetch=eval_kwargs.get('prefetch', False),
            img_norm_cfg=eval_kwargs.get('img_norm_cfg', dict()),
        )
        return dataset, data_loader

    def _build_metric_tools(self, metric_args):
        if self.metric_mode == "knn":
            self.metric = WeightedKNNClassifier(
                k=metric_args.get('knn', 20),
                T=metric_args.get('temperature', 0.07),
                distance_fx=metric_args.get('metric', 'cosine'),
                epsilon=metric_args.get('epsilon', 1e-5),
            )
        else:
            self.metric = None

    def _build_visualization_tools(self, visual_args):
        from sklearn.manifold import TSNE
        try:
            import umap
        except:
            self.visual_mode = "tsne"
            print("Could not import `umap`, using `tsne` instead!")
        if self.visual_mode == "umap":
            self.visualize = umap.UMAP(
                n_neighbors=visual_args.get('n_neighbors', 15),
                n_components=visual_args.get('n_components', 2),
                n_epochs=visual_args.get('n_epochs', 200),
                learning_rate=visual_args.get('learning_rate', 1.0),
                metric=visual_args.get('metric', 'euclidean'),
                init=visual_args.get('init', 'spectral'),
            )
        elif self.visual_mode == "tsne":
            self.visualize = TSNE(
                n_components=visual_args.get('n_components', 2),
                perplexity=visual_args.get('perplexity', 30),
                early_exaggeration=visual_args.get('early_exaggeration', 12.0),
                learning_rate=visual_args.get('learning_rate', 200.0),
                n_iter=visual_args.get('n_iter', 1000),
                n_iter_without_progress=visual_args.get('n_iter_without_progress', 300),
                metric=visual_args.get('metric', 'euclidean'),
                init=visual_args.get('init', 'random'),
            )
        else:
            self.visualize = None

    def _plot_visualization(self, results, labels, save_name):
        res_min, res_max = results.min(0), results.max(0)
        res_norm = (results - res_min) / (res_max - res_min)
        plt.figure(figsize=(10, 10))
        plt.scatter(
            res_norm[:, 0], res_norm[:, 1],
            alpha=self.visual_args.get('plot_alpha', 1.0),
            s=self.visual_args.get('plot_s', 15),
            c=labels,
            cmap=self.visual_args.get('plot_cmap', 'tab20'))
        plt.savefig(save_name)
        plt.close()

    def before_run(self, runner):
        # save dirs
        if self.save_val:
            mmcv.mkdir_or_exist(f'{runner.work_dir}/metric/')
        if self.visualize is not None:
            mmcv.mkdir_or_exist(f'{runner.work_dir}/visualization/')
        if self.initial:
            self._run_validate(runner)

    def after_train_epoch(self, runner):
        if not self.every_n_epochs(runner, self.interval):
            return
        self._run_validate(runner)

    def _run_validate(self, runner):
        runner.model.eval()
        func = lambda **x: runner.model(mode=self.forward_mode, **x)
        if self.dist_mode:
            if self.metric is not None:
                train_results = dist_forward_collect(  # dict{key: np.ndarray}
                    func, self.train_data_loader, runner.rank, len(self.train_dataset))
            val_results = dist_forward_collect(  # dict{key: np.ndarray}
                func, self.val_data_loader, runner.rank, len(self.val_dataset))
        else:
            if self.metric is not None:
                train_results = nondist_forward_collect(
                    func, self.train_data_loader, len(self.train_dataset))
            val_results = nondist_forward_collect(
                func, self.val_data_loader, len(self.val_dataset))
        if runner.rank == 0:
            # train and evaluate metric
            if self.metric is not None:
                for name in val_results.keys():
                    eval_res = self.metric.evaluate(
                        torch.from_numpy(train_results[name]),
                        torch.from_numpy(self.train_dataset.targets),
                        torch.from_numpy(val_results[name]),
                        torch.from_numpy(self.val_dataset.targets),
                        keyword=name, logger=runner.logger,
                        **self.eval_kwargs['eval_param'])
                    for key, val in eval_res.items():
                        runner.log_buffer.output[key] = val
                    runner.log_buffer.ready = True
                    if self.save_val:
                        np.save(f"{runner.work_dir}/metric/val_epoch_{runner.epoch}.npy",
                                val_results[name])
            # visualization
            if self.visualize is not None:
                for name, val in val_results.items():
                    val = self.visualize.fit_transform(val)
                    self._plot_visualization(
                        val, self.val_dataset.targets,
                        f"{runner.work_dir}/visualization/{name}_epoch_{runner.epoch}.png")
                    if self.save_val:
                        np.save(f"{runner.work_dir}/metric/val_{name}_2d_epoch_{runner.epoch}.npy", val)
        runner.model.train()
