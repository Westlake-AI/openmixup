# This file is modified from
# https://github.com/vturrisi/solo-learn/solo/utils/knn.py
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F

from openmixup.utils import print_log


class WeightedKNNClassifier():
    """Implements the weighted k-NN classifier for evaluation.

    KNN metric is propised in "Unsupervised Feature Learning via Non-Parametric
    Instance Discrimination (https://arxiv.org/pdf/1805.01978.pdf)"
        https://github.com/zhirongw/lemniscate.pytorch

    Args:
        k (int, optional): number of neighbors. Defaults to 20.
        T (float, optional): temperature for the exponential. Only used with cosine
            distance. Defaults to 0.07.
        chunk_size (int, optional): Mini batch size for performing knn classification.
            Reduce the chunk_size when the number of train samples is too large, which
            might cause the distance matrix out of CUDA memory. Defaults to 128.
        distance_fx (str, optional): Distance function. Accepted arguments: "cosine" or
            "euclidean". Defaults to "cosine".
        epsilon (float, optional): Small value for numerical stability. Only used with
            euclidean distance. Defaults to 1e-5.
    """

    def __init__(self,
                 k=20,
                 T=0.07,
                 chunk_size=128,
                 distance_fx="cosine",
                 epsilon=1e-5,
                ):
        super().__init__()
        self.k = k
        self.T = T
        self.chunk_size = chunk_size
        self.distance_fx = distance_fx
        self.epsilon = epsilon
        self.model_path = None
        self.save_model = False

    @torch.no_grad()
    def evaluate(self,
                 train_features, train_targets, test_features, test_targets,
                 keyword, logger=None, topk=(1, 5), **kwargs):
        """Computes weighted k-NN accuracy top-1 & top-5.
        
        If cosine distance is selected, the weight is computed using the exponential
        of the temperature scaled cosine distance of the samples. If euclidean distance
        is selected, the weight corresponds to the inverse of the euclidean distance.

        Args:
            train_features (torch.Tensor | np.array): Train features in (N,D).
            train_targets (torch.Tensor | np.array): Train targets in (N,C).
            test_features (torch.Tensor | np.array): Test features in (N,D).
            test_targets (torch.Tensor | np.array): Test targets in (N,C).
            logger (logging.Logger | str | None, optional): The defined logger
                to be used. Defaults to None.
            topk (tuple(int)): The output includes topk accuracy.
        """
        eval_res = {}

        to_tensor = (lambda x: torch.from_numpy(x)
                    if isinstance(x, np.ndarray) else x)
        train_features = to_tensor(train_features)
        train_targets = to_tensor(train_targets)
        test_features = to_tensor(test_features)
        test_targets = to_tensor(test_targets)

        if self.distance_fx == "cosine":
            train_features = F.normalize(train_features)
            test_features = F.normalize(test_features)

        num_classes = torch.unique(test_targets).numel()
        num_train_images = train_targets.size(0)
        num_test_images = test_targets.size(0)
        num_train_images = train_targets.size(0)
        chunk_size = min(self.chunk_size, num_test_images)
        k = min(self.k, num_train_images)

        top1, top5, total = 0.0, 0.0, 0
        retrieval_one_hot = torch.zeros(k, num_classes).to(train_features.device)
        for idx in tqdm(range(0, num_test_images, chunk_size)):
            # get the features for test images
            features = test_features[idx : min((idx + chunk_size), num_test_images), :]
            targets = test_targets[idx : min((idx + chunk_size), num_test_images)]
            batch_size = targets.size(0)

            # calculate the dot product and compute top-k neighbors
            if self.distance_fx == "cosine":
                similarities = torch.mm(features, train_features.t())
            elif self.distance_fx == "euclidean":
                similarities = 1 / (torch.cdist(features, train_features) + self.epsilon)
            else:
                raise NotImplementedError

            similarities, indices = similarities.topk(k, largest=True, sorted=True)
            candidates = train_targets.view(1, -1).expand(batch_size, -1)
            retrieved_neighbors = torch.gather(candidates, 1, indices)

            retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
            retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)

            if self.distance_fx == "cosine":
                similarities = similarities.clone().div_(self.T).exp_()

            probs = torch.sum(
                torch.mul(retrieval_one_hot.view(batch_size, -1, num_classes),
                         similarities.view(batch_size, -1, 1)),
                1,
            )
            _, predictions = probs.sort(1, True)

            # find the predictions that match the target
            correct = predictions.eq(targets.data.view(-1, 1))
            top1 = top1 + correct.narrow(1, 0, 1).sum().item()
            top5 = (
                top5 + correct.narrow(1, 0, min(5, k, correct.size(-1))).sum().item()
            )  # top5 does not make sense if k < 5
            total += targets.size(0)

        eval_res["{}_knn_top1".format(keyword)] = top1 * 100.0 / total
        if 5 in topk:
            eval_res["{}_knn_top5".format(keyword)] = top5 * 100.0 / total
        if logger is not None and logger != 'silent':
            for k,v in eval_res.items():
                print_log("{}: {:.03f}".format(k, v), logger=logger)

        return eval_res
