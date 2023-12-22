from numbers import Number

import torch
import torch.nn as nn


def accuracy(pred, target, topk=1, thrs=0.):
    """Calculate accuracy according to the prediction and target.

    Args:
        pred (torch.Tensor): The model prediction.
        target (torch.Tensor): The target of each prediction
        topk (int | tuple[int]): If the predictions in ``topk``
            matches the target, the predictions will be regarded as
            correct ones. Defaults to 1.
        thrs (Number, optional): Predictions with scores under
            the thresholds are considered negative. Default to 0.

    Returns:
        torch.Tensor | list[torch.Tensor] | list[list[torch.Tensor]]: Accuracy
            - torch.Tensor: If both ``topk`` and ``thrs`` is a single value.
            - list[torch.Tensor]: If one of ``topk`` or ``thrs`` is a tuple.
            - list[list[torch.Tensor]]: If both ``topk`` and ``thrs`` is a \
              tuple. And the first dim is ``topk``, the second dim is ``thrs``.
    """
    assert isinstance(topk, (int, tuple)), \
        f'topk should be a number or tuple, but got {type(topk)}.'
    assert isinstance(thrs, Number), \
        f'thrs should be a number, but got {type(thrs)}.'
    if isinstance(topk, int):
        topk = (topk, )
        return_single = True
    else:
        return_single = False

    maxk = max(topk)
    pred_score, pred_label = pred.topk(maxk, dim=1)
    pred_label = pred_label.t()
    correct = pred_label.eq(target.view(1, -1).expand_as(pred_label))

    res = []
    for k in topk:
        if thrs > 0.:
            # Only prediction values larger than thr are counted as correct
            _correct = correct & (pred_score.t() > thrs)
            correct_k = _correct[:k].reshape(-1).float().sum(0, keepdim=True)
        else:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / pred.size(0)))
    return res[0] if return_single else res


def accuracy_mixup(pred, targets):
    """ Accuracy for mixup classification 
    
    Args:
        pred (tensor): Nxd predictions.
        targets (tuple): Mixup labels tuple (y_a, y_b, lambda).
    Return:
        Res: Single result of lam * Acc_lam + (1-lam) * Acc_lam_.
    """
    lam = targets[2]
    if isinstance(lam, type(pred)):  # bugs of sample-wise lam
        lam = lam.mean().cpu().numpy()
    maxk = 2  # top-2
    _, pred_label = pred.topk(maxk, dim=1)
    pred_label = pred_label.t()  # 2xN
    # assumpting lam > 0.5
    if lam > 0.5:
        y_a, y_b, lam = targets
        lam_ = 1 - lam
    else:
        y_b, y_a, lam_ = targets
        lam = 1 - lam_
    # top-1 for lam
    correct_y_a = pred_label.eq(y_a.view(1, -1).expand_as(pred_label))
    correct_lam = correct_y_a[:1].view(-1).float().sum(0, keepdim=True)
    res_lam = correct_lam.mul_(100.0 / pred.size(0))
    # top-2 for lam_
    correct_y_b = pred_label.eq(y_b.view(1, -1).expand_as(pred_label))
    correct_lam_ = correct_y_b[1:].view(-1).float().sum(0, keepdim=True)
    res_lam_ = correct_lam_.mul_(100.0 / pred.size(0))

    return res_lam * lam + res_lam_ * lam_


def accuracy_co_mixup(pred, targets):
    """ Accuracy for mixup classification with Co-Mixup """
    lam = targets[-1]
    y = targets[:-1]
    maxk = 3  # top-2
    _, pred_label = pred.topk(maxk, dim=1)
    pred_label = pred_label.t()  # 3xN

    res_lam = []
    # top-1 for lam_a
    for i in range(0, len(y)):
        correct_y = pred_label.eq(y[i].view(1, -1).expand_as(pred_label))
        if i == len(y)-1:
            correct_lam = correct_y[i:].view(-1).float().sum(0, keepdim=True)
        else:
            correct_lam = correct_y[i:i+1].view(-1).float().sum(0, keepdim=True)
        res_lam.append(correct_lam.mul_(100.0 / pred.size(0)))

    total_res = 0.0
    for i in range(0, len(res_lam)):
        total_res += res_lam[i]*lam[i]

    return total_res


def accuracy_semantic_softmax(pred, target, processor):
    """Forward function to calculate the semantic softmax accuracy.

    Args:
        pred (torch.Tensor): Prediction of models.
        target (torch.Tensor): Target for each prediction.

    Returns:
        list[torch.Tensor]: The accuracies under different topk criterions.
    """
    with torch.no_grad():
        semantic_logit_list = processor.split_logits_to_semantic_logits(pred)
        semantic_targets = processor.convert_targets_to_semantic_targets(target)
        accuracy_list = []
        accuracy_valid_list = []
        result = 0
        for i in range(len(semantic_logit_list)):  # scanning hirarchy_level_list
            logits_i = semantic_logit_list[i]
            targets_i = semantic_targets[:, i]
            pred_i = logits_i.argmax(dim=-1)
            ind_valid = (targets_i >= 0)
            num_valids = torch.sum(ind_valid)
            accuracy_valid_list.append(num_valids)
            if num_valids > 0:
                accuracy_list.append((
                    pred_i[ind_valid] == targets_i[ind_valid]).float().mean())
            else:
                accuracy_list.append(0)
            result += accuracy_list[-1] * accuracy_valid_list[-1]
        num_valids_total = sum(accuracy_valid_list)

    return [result / num_valids_total * 100]


class Accuracy(nn.Module):

    def __init__(self, topk=(1, )):
        """Module to calculate the accuracy.

        Args:
            topk (tuple): The criterion used to calculate the
                accuracy. Defaults to (1,).
        """
        super().__init__()
        self.topk = topk

    def forward(self, pred, target):
        """Forward function to calculate accuracy.

        Args:
            pred (torch.Tensor): Prediction of models.
            target (torch.Tensor): Target for each prediction.

        Returns:
            list[torch.Tensor]: The accuracies under different topk criterions.
        """
        return accuracy(pred, target, self.topk)
