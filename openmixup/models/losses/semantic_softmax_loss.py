import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import LOSSES
from .utils import weight_reduce_loss


@torch.jit.script
def stable_softmax(logits: torch.Tensor):
    logits_m = logits - logits.max(dim=1)[0].unsqueeze(1)
    exp = torch.exp(logits_m)
    probs = exp / torch.sum(exp, dim=1).unsqueeze(1)
    return probs


class ImageNet21kSemanticSoftmax:
    r"""SemanticSoftmax Preprocesser for ImageNet-21K.

    ImageNet-21K Pretraining for the Masses. In NIPS, 2021.
    <https://arxiv.org/abs/2104.10972>

    Args:
        tree_path (str): The path to the semantic file for ImageNet-21K.
    """

    def __init__(self, tree_path):
        try:
            self.tree = torch.load(tree_path)
            self.class_tree_list = self.tree['class_tree_list']
            self.class_names = np.array(list(self.tree['class_description'].values()))
        except:
            raise RuntimeError("Invalid semantic tree.")
        self.max_normalization_factor = 20.
        num_classes = len(self.class_tree_list)
        self.class_depth = torch.zeros(num_classes)
        for i in range(num_classes):
            self.class_depth[i] = len(self.class_tree_list[i]) - 1
        max_depth = int(torch.max(self.class_depth).item())

        # process semantic relations
        hist_tree = torch.histc(
            self.class_depth, bins=max_depth + 1, min=0, max=max_depth).int()
        ind_list = []
        class_names_ind_list = []
        hirarchy_level_list = []
        cls = torch.tensor(np.arange(num_classes))
        for i in range(max_depth):
            if hist_tree[i] > 1:
                hirarchy_level_list.append(i)
                ind_list.append(cls[self.class_depth == i].long())
                class_names_ind_list.append(self.class_names[ind_list[-1]])
        self.hierarchy_indices_list = ind_list
        self.hirarchy_level_list = hirarchy_level_list
        self.class_names_ind_list = class_names_ind_list

        # calcuilating normalization array
        self.normalization_factor_list = torch.zeros_like(hist_tree)
        self.normalization_factor_list[-1] = hist_tree[-1]
        for i in range(max_depth):
            self.normalization_factor_list[i] = torch.sum(hist_tree[i:], dim=0)
        self.normalization_factor_list = \
            self.normalization_factor_list[0] / self.normalization_factor_list
        if self.max_normalization_factor:
            self.normalization_factor_list.clamp_(max=self.max_normalization_factor)

    def split_logits_to_semantic_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """
        split logits to 11 different hierarchies.

        :param self.self.hierarchy_indices_list: a list of size [num_of_hierarchies].
        Each element in the list is a tensor that contains the corresponding indices
            for the relevant hierarchy
        """
        semantic_logit_list = []
        for i, ind in enumerate(self.hierarchy_indices_list):
            logits_i = logits[:, ind]
            semantic_logit_list.append(logits_i)
        return semantic_logit_list

    def convert_targets_to_semantic_targets(self,
                                            targets_original: torch.Tensor) -> torch.Tensor:
        """
        converts single-label targets to targets over num_of_hierarchies hierarchies.
        [batch_size] -> [batch_size x num_of_hierarchies].
        if no hierarchical target is available, outputs -1.

        :param self.self.hierarchy_indices_list: a list of size [num_of_hierarchies].
        Each element in the list is a tensor that contains the corresponding indices
            for the relevant hierarchy

        :param self.class_tree_list: a list of size [num_of_classes].
        Each element in the list is a list of the relevent sub-hirrachies.

        Example - self.class_tree_list[10]:  [10, 9, 66, 65, 144]
        """
        targets = targets_original.cpu().detach()  # dont edit original targets
        semantic_targets_list = torch.zeros((
            targets.shape[0], len(self.hierarchy_indices_list))) - 1
        for i, target in enumerate(targets.cpu()):  # scanning over batch size
            cls_multi_list = self.class_tree_list[target]  # all the sub-hirrachies of the rager
            hir_levels = len(cls_multi_list)
            for j, cls in enumerate(cls_multi_list):
                # protection for too small hirarchy_level_list.
                # this protection enables us to remove hierarchies
                if len(self.hierarchy_indices_list) <= hir_levels - j - 1:
                    continue
                ind_valid = (self.hierarchy_indices_list[hir_levels - j - 1] == cls)
                semantic_targets_list[i, hir_levels - j - 1] = torch.where(ind_valid)[0]

        return semantic_targets_list.long().to(device=targets_original.device)

    def estimate_teacher_confidence(self, preds_teacher: torch.Tensor) -> torch.Tensor:
        """
        Helper function:
        return the sum probabilities of the top 5% classes in preds_teacher.
        preds_teacher dimensions - [batch_size x num_of_classes]
        """
        with torch.no_grad():
            num_elements = preds_teacher.shape[1]
            num_elements_topk = int(np.ceil(num_elements / 20))  # top 5%
            weights_batch = torch.sum(torch.topk(preds_teacher, num_elements_topk).values, dim=1)
        return weights_batch

    def calculate_KD_loss(self, input_student: torch.Tensor, input_teacher: torch.Tensor):
        """
        Calculates the semantic KD-MSE distance between student and teacher probabilities
        input_student dimensions - [batch_size x num_of_classes]
        input_teacher dimensions - [batch_size x num_of_classes]
        """

        semantic_input_student = self.split_logits_to_semantic_logits(input_student)
        semantic_input_teacher = self.split_logits_to_semantic_logits(input_teacher)
        number_of_hierarchies = len(semantic_input_student)

        losses_list = []
        # scanning hirarchy_level_list
        for i in range(number_of_hierarchies):
            # converting to semantic logits
            inputs_student_i = semantic_input_student[i]
            inputs_teacher_i = semantic_input_teacher[i]

            # generating probs
            preds_student_i = stable_softmax(inputs_student_i)
            preds_teacher_i = stable_softmax(inputs_teacher_i)

            # weight MSE-KD distances according to teacher confidence
            loss_non_reduced = torch.nn.MSELoss(reduction='none')(preds_student_i, preds_teacher_i)
            weights_batch = self.estimate_teacher_confidence(preds_teacher_i)
            loss_weighted = loss_non_reduced * weights_batch.unsqueeze(1)
            losses_list.append(torch.sum(loss_weighted))

        return sum(losses_list)


@LOSSES.register_module()
class SemanticSoftmaxLoss(nn.Module):
    r"""SemanticSoftmax Cross Entropy Loss.

    ImageNet-21K Pretraining for the Masses. In NIPS, 2021.
    <https://arxiv.org/abs/2104.10972>

    Args:
        processor (dict): The name of the semantic softmax processor.
        label_smooth_val (float): The degree of label smoothing.
        reduction (str): The method used to reduce the loss.
            Options are "none", "mean" and "sum". Defaults to 'mean'.
        loss_weight (float):  Weight of the loss. Defaults to 1.0.
    """

    def __init__(self,
                 processor='imagenet21k',
                 tree_path=None,
                 label_smooth_val=0.1,
                 reduction='mean',
                 loss_weight=1.0,
                 **kwargs):
        super(SemanticSoftmaxLoss, self).__init__()
        self.processor = ImageNet21kSemanticSoftmax(tree_path) \
            if processor == 'imagenet21k' else None
        assert (isinstance(label_smooth_val, float)
                and 0 <= label_smooth_val < 1), \
            f'LabelSmoothLoss accepts a float label_smooth_val ' \
            f'over [0, 1), but gets {label_smooth_val}'
        self.label_smooth_val = label_smooth_val

        self.reduction = reduction
        self.loss_weight = loss_weight
        self.post_process = "none"

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        r"""caculate loss
        
        Args:
            cls_score (tensor): Predicted logits of (N, C).
            label (tensor): Groundtruth label of (N, \*).
            weight (tensor): Loss weight for each samples of (N,).
            avg_factor (int, optional): Average factor that is used to average the loss.
                Defaults to None.
            reduction_override (str, optional): The reduction method used to override
                the original reduction method of the loss. Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override is not None else self.reduction)

        semantic_logit_list = self.processor.split_logits_to_semantic_logits(cls_score)
        semantic_targets_tensor = self.processor.convert_targets_to_semantic_targets(label)

        losses_list = []
        # scanning hirarchy_level_list
        for i in range(len(semantic_logit_list)):
            logits_i = semantic_logit_list[i]
            targets_i = semantic_targets_tensor[:, i]

            # generate probs
            log_preds = F.log_softmax(logits_i, dim=1)

            # generate targets (with protections)
            targets_i_valid = targets_i.clone()
            targets_i_valid[targets_i_valid < 0] = 0
            num_classes = logits_i.size()[-1]
            targets_classes = torch.zeros_like(logits_i
                                               ).scatter_(1, targets_i_valid.unsqueeze(1), 1)
            targets_classes.mul_(1 - self.label_smooth_val).add_(self.label_smooth_val / num_classes)

            cross_entropy_loss_tot = -targets_classes.mul(log_preds)
            cross_entropy_loss_tot *= ((targets_i >= 0).unsqueeze(1))
            loss_i = cross_entropy_loss_tot.sum(dim=-1)  # sum over classes
            if avg_factor is None:
                if reduction == 'mean':
                    loss_i = loss_i.mean()  # mean over batch
            else:
                loss_i = weight_reduce_loss(loss_i, weight, reduction, avg_factor)
            losses_list.append(loss_i)

        total_sum = 0.
        for i, loss_h in enumerate(losses_list):  # summing over hirarchies
            total_sum += loss_h * self.processor.normalization_factor_list[i]

        return total_sum
