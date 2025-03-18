from numbers import Number

import numpy as np
from scipy import stats
import torch
from torch.nn.functional import one_hot


def calculate_confusion_matrix(pred, target):
    """Calculate confusion matrix according to the prediction and target.

    Args:
        pred (torch.Tensor | np.array): The model prediction with shape (N, C).
        target (torch.Tensor | np.array): The target of each prediction with
            shape (N, 1) or (N,).

    Returns:
        torch.Tensor: Confusion matrix
            The shape is (C, C), where C is the number of classes.
    """

    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target)
    assert (
        isinstance(pred, torch.Tensor) and isinstance(target, torch.Tensor)), \
        (f'pred and target should be torch.Tensor or np.ndarray, '
         f'but got {type(pred)} and {type(target)}.')

    # Modified from PyTorch-Ignite
    num_classes = pred.size(1)
    pred_label = torch.argmax(pred, dim=1).flatten()
    target_label = target.flatten()
    assert len(pred_label) == len(target_label)

    with torch.no_grad():
        indices = num_classes * target_label + pred_label
        matrix = torch.bincount(indices, minlength=num_classes**2)
        matrix = matrix.reshape(num_classes, num_classes)
    return matrix


def precision_recall_f1(pred, target, average_mode='macro', thrs=0.):
    """Calculate precision, recall and f1 score according to the prediction and
    target.

    Args:
        pred (torch.Tensor | np.array): The model prediction with shape (N, C).
        target (torch.Tensor | np.array): The target of each prediction with
            shape (N, 1) or (N,).
        average_mode (str): The type of averaging performed on the result.
            Options are 'macro' and 'none'. If 'none', the scores for each
            class are returned. If 'macro', calculate metrics for each class,
            and find their unweighted mean.
            Defaults to 'macro'.
        thrs (Number | tuple[Number], optional): Predictions with scores under
            the thresholds are considered negative. Default to 0.

    Returns:
        tuple: tuple containing precision, recall, f1 score.

            The type of precision, recall, f1 score is one of the following:

        +----------------------------+--------------------+-------------------+
        | Args                       | ``thrs`` is number | ``thrs`` is tuple |
        +============================+====================+===================+
        | ``average_mode`` = "macro" | float              | list[float]       |
        +----------------------------+--------------------+-------------------+
        | ``average_mode`` = "none"  | np.array           | list[np.array]    |
        +----------------------------+--------------------+-------------------+
    """

    allowed_average_mode = ['macro', 'none']
    if average_mode not in allowed_average_mode:
        raise ValueError(f'Unsupport type of averaging {average_mode}.')

    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)
    assert isinstance(pred, torch.Tensor), \
        (f'pred should be torch.Tensor or np.ndarray, but got {type(pred)}.')
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target).long()
    assert isinstance(target, torch.Tensor), \
        f'target should be torch.Tensor or np.ndarray, ' \
        f'but got {type(target)}.'

    if isinstance(thrs, Number):
        thrs = (thrs, )
        return_single = True
    elif isinstance(thrs, tuple):
        return_single = False
    else:
        raise TypeError(
            f'thrs should be a number or tuple, but got {type(thrs)}.')

    num_classes = pred.size(1)
    pred_score, pred_label = torch.topk(pred, k=1)
    pred_score = pred_score.flatten()
    pred_label = pred_label.flatten()

    gt_positive = one_hot(target.flatten(), num_classes)

    precisions = []
    recalls = []
    f1_scores = []
    for thr in thrs:
        # Only prediction values larger than thr are counted as positive
        pred_positive = one_hot(pred_label, num_classes)
        if thr is not None:
            pred_positive[pred_score <= thr] = 0
        class_correct = (pred_positive & gt_positive).sum(0)
        precision = class_correct / np.maximum(pred_positive.sum(0), 1.) * 100
        recall = class_correct / np.maximum(gt_positive.sum(0), 1.) * 100
        f1_score = 2 * precision * recall / np.maximum(
            precision + recall,
            torch.finfo(torch.float32).eps)
        if average_mode == 'macro':
            precision = float(precision.mean())
            recall = float(recall.mean())
            f1_score = float(f1_score.mean())
        elif average_mode == 'none':
            precision = precision.detach().cpu().numpy()
            recall = recall.detach().cpu().numpy()
            f1_score = f1_score.detach().cpu().numpy()
        else:
            raise ValueError(f'Unsupport type of averaging {average_mode}.')
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)

    if return_single:
<<<<<<< HEAD
        return precisions[0], recalls[0], f1_scores[0]
    else:
        return precisions, recalls, f1_scores
=======
        return (precisions[0], recalls[0], f1_scores[0])
    else:
        return (precisions, recalls, f1_scores)
>>>>>>> db2c4ac (update some vit-based mixup methods and fix robustness eval tasks)


def precision(pred, target, average_mode='macro', thrs=0.):
    """Calculate precision according to the prediction and target.

    Args:
        pred (torch.Tensor | np.array): The model prediction with shape (N, C).
        target (torch.Tensor | np.array): The target of each prediction with
            shape (N, 1) or (N,).
        average_mode (str): The type of averaging performed on the result.
            Options are 'macro' and 'none'. If 'none', the scores for each
            class are returned. If 'macro', calculate metrics for each class,
            and find their unweighted mean.
            Defaults to 'macro'.
        thrs (Number | tuple[Number], optional): Predictions with scores under
            the thresholds are considered negative. Default to 0.

    Returns:
         float | np.array | list[float | np.array]: Precision.

        +----------------------------+--------------------+-------------------+
        | Args                       | ``thrs`` is number | ``thrs`` is tuple |
        +============================+====================+===================+
        | ``average_mode`` = "macro" | float              | list[float]       |
        +----------------------------+--------------------+-------------------+
        | ``average_mode`` = "none"  | np.array           | list[np.array]    |
        +----------------------------+--------------------+-------------------+
    """
    precisions, _, _ = precision_recall_f1(pred, target, average_mode, thrs)
    return precisions


def recall(pred, target, average_mode='macro', thrs=0.):
    """Calculate recall according to the prediction and target.

    Args:
        pred (torch.Tensor | np.array): The model prediction with shape (N, C).
        target (torch.Tensor | np.array): The target of each prediction with
            shape (N, 1) or (N,).
        average_mode (str): The type of averaging performed on the result.
            Options are 'macro' and 'none'. If 'none', the scores for each
            class are returned. If 'macro', calculate metrics for each class,
            and find their unweighted mean.
            Defaults to 'macro'.
        thrs (Number | tuple[Number], optional): Predictions with scores under
            the thresholds are considered negative. Default to 0.

    Returns:
         float | np.array | list[float | np.array]: Recall.

        +----------------------------+--------------------+-------------------+
        | Args                       | ``thrs`` is number | ``thrs`` is tuple |
        +============================+====================+===================+
        | ``average_mode`` = "macro" | float              | list[float]       |
        +----------------------------+--------------------+-------------------+
        | ``average_mode`` = "none"  | np.array           | list[np.array]    |
        +----------------------------+--------------------+-------------------+
    """
    _, recalls, _ = precision_recall_f1(pred, target, average_mode, thrs)
    return recalls


def f1_score(pred, target, average_mode='macro', thrs=0.):
    """Calculate F1 score according to the prediction and target.

    Args:
        pred (torch.Tensor | np.array): The model prediction with shape (N, C).
        target (torch.Tensor | np.array): The target of each prediction with
            shape (N, 1) or (N,).
        average_mode (str): The type of averaging performed on the result.
            Options are 'macro' and 'none'. If 'none', the scores for each
            class are returned. If 'macro', calculate metrics for each class,
            and find their unweighted mean.
            Defaults to 'macro'.
        thrs (Number | tuple[Number], optional): Predictions with scores under
            the thresholds are considered negative. Default to 0.

    Returns:
         float | np.array | list[float | np.array]: F1 score.

        +----------------------------+--------------------+-------------------+
        | Args                       | ``thrs`` is number | ``thrs`` is tuple |
        +============================+====================+===================+
        | ``average_mode`` = "macro" | float              | list[float]       |
        +----------------------------+--------------------+-------------------+
        | ``average_mode`` = "none"  | np.array           | list[np.array]    |
        +----------------------------+--------------------+-------------------+
    """
    _, _, f1_scores = precision_recall_f1(pred, target, average_mode, thrs)
    return f1_scores


def support(pred, target, average_mode='macro'):
    """Calculate the total number of occurrences of each label according to the
    prediction and target.

    Args:
        pred (torch.Tensor | np.array): The model prediction with shape (N, C).
        target (torch.Tensor | np.array): The target of each prediction with
            shape (N, 1) or (N,).
        average_mode (str): The type of averaging performed on the result.
            Options are 'macro' and 'none'. If 'none', the scores for each
            class are returned. If 'macro', calculate metrics for each class,
            and find their unweighted sum.
            Defaults to 'macro'.

    Returns:
        float | np.array: Support.

            - If the ``average_mode`` is set to macro, the function returns
              a single float.
            - If the ``average_mode`` is set to none, the function returns
              a np.array with shape C.
    """
    confusion_matrix = calculate_confusion_matrix(pred, target)
    with torch.no_grad():
        res = confusion_matrix.sum(1)
        if average_mode == 'macro':
            res = float(res.sum().numpy())
        elif average_mode == 'none':
            res = res.numpy()
        else:
            raise ValueError(f'Unsupport type of averaging {average_mode}.')
    return res


def regression_error(pred, target, average_mode='mean'):
    """Calculate mean squared error (MSE) and mean absolute error (MAE).

    Args:
        pred (torch.Tensor | np.array): The model prediction with shape (N, \*).
        target (torch.Tensor | np.array): The target of each prediction with
            shape (N, \*), which should be normalized.
        average_mode (str): The type of averaging performed on the result.
            Options are 'mean' and 'none'. If 'none', the sum of error will be
            returned. If 'mean', calculate mean of error. Defaults to 'mean'.

    Returns:
        tuple: tuple containing MSE, MAE.
    """

    allowed_average_mode = ['mean', 'none']
    if average_mode not in allowed_average_mode:
        raise ValueError(f'Unsupport type of averaging {average_mode}.')

    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)
    assert isinstance(pred, torch.Tensor), \
        (f'pred should be torch.Tensor or np.ndarray, but got {type(pred)}.')
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target).long()
    assert isinstance(target, torch.Tensor), \
        f'target should be torch.Tensor or np.ndarray, ' \
        f'but got {type(target)}.'

    mse = torch.square(pred - target).sum()
    mae = torch.abs(pred - target).sum()
<<<<<<< HEAD
    if average_mode == 'mean':
        mse /= pred.size(0)
        mae /= pred.size(0)

    return mse, mae
=======
    mape = torch.sum(torch.abs(pred - target) / torch.clamp(torch.abs(target), min=1e-8))
    if average_mode == 'mean':
        mse /= pred.size(0)
        mae /= pred.size(0)
        mape /= pred.size(0)
    rmse = torch.sqrt(mse)

    return (mse, mae, rmse, mape)
>>>>>>> db2c4ac (update some vit-based mixup methods and fix robustness eval tasks)


def pearson_correlation(pred, target, average_mode='mean'):
    """Calculate Pearson Correlation.

    Args:
        pred (torch.Tensor | np.array): The model prediction with shape (N, \*).
        target (torch.Tensor | np.array): The target of each prediction with
            shape (N, \*), which should be normalized.
        average_mode (str): The type of averaging performed on the result.
            Options are 'mean' and 'none'. If 'none', the sum of error will be
            returned. If 'mean', calculate mean of error. Defaults to 'mean'.

    Returns:
        float: correlation.
    """
    allowed_average_mode = ['mean', 'none', None]
    if average_mode not in allowed_average_mode:
        raise ValueError(f'Unsupport type of averaging {average_mode}.')
    
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)
    assert isinstance(pred, torch.Tensor), \
        (f'pred should be torch.Tensor or np.ndarray, but got {type(pred)}.')
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target).long()
    assert isinstance(target, torch.Tensor), \
        f'target should be torch.Tensor or np.ndarray, ' \
        f'but got {type(target)}.'
    
    x = pred.view(pred.size(0), -1)
    y = target.view(target.size(0), -1)
    vx = x - x.mean()
    vy = y - y.mean()
    corr = (vx * vy).sum() / ((vx ** 2).sum().sqrt() * (vy ** 2).sum().sqrt() + 1e-20)
    if average_mode == 'mean':
        corr = corr.mean()

    return corr


def spearman_correlation(pred, target, average_mode='mean'):
    """Calculate Spearman Correlation with scipy.

    Args:
        pred (torch.Tensor | np.array): The model prediction with shape (N, \*).
        target (torch.Tensor | np.array): The target of each prediction with
            shape (N, \*), which should be normalized.
        average_mode (str): The type of averaging performed on the result.
            Options are 'mean' and 'none'. If 'none', the sum of error will be
            returned. If 'mean', calculate mean of error. Defaults to 'mean'.

    Returns:
        float: correlation.
    """
    allowed_average_mode = ['mean', 'none', None]
    if average_mode not in allowed_average_mode:
        raise ValueError(f'Unsupport type of averaging {average_mode}.')
    
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    assert isinstance(pred, np.ndarray), \
        (f'pred should be torch.Tensor or np.ndarray, but got {type(pred)}.')
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    assert isinstance(target, np.ndarray), \
        f'target should be torch.Tensor or np.ndarray, ' \
        f'but got {type(target)}.'

    pred = pred.reshape(pred.shape[0], -1)
    target = target.reshape(target.shape[0], -1)
    if average_mode == "mean":
        corr, _ = stats.spearmanr(pred, target, axis=None)
    else:
        corr, _ = stats.spearmanr(pred, target, axis=0)

    return corr
