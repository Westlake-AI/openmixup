# This file is modified from
# https://github.com/facebookresearch/fair_self_supervision_benchmark

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import json
import logging
import multiprocessing as mp
import os.path as osp
import pickle
import sys
import six

import numpy as np
import mmcv
import torch
from torch.nn.functional import one_hot
from tqdm import tqdm
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC

from openmixup.utils import print_log


class SVMHelper():
    """Helper module for svm training and testing"""

    def __init__(self):
        pass

    @staticmethod
    def logger():
        # create the logger
        FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
        logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
        logger = logging.getLogger(__name__)
        return logger

    @staticmethod
    def py2_py3_compatible_cost(cost):
        """Python 2 and python 3 have different floating point precision.

        The following trick helps keep the backwards compatibility.
        """
        return str(float(f'{cost:.17f}'))

    @staticmethod
    def load_json(file_path):
        """Load json file."""
        assert osp.exists(file_path), f'{file_path} does not exist'
        with open(file_path, 'r') as fp:
            data = json.load(fp)
        img_ids = list(data.keys())
        cls_names = list(data[img_ids[0]].keys())
        return img_ids, cls_names

    @staticmethod
    def get_svm_train_output_files(cls, cost, output_path):
        """Get output file path."""
        cls_cost = str(cls) + '_cost' + SVMHelper.py2_py3_compatible_cost(cost)
        out_file = osp.join(output_path, 'cls' + cls_cost + '.pickle')
        ap_matrix_out_file = osp.join(output_path, 'AP_cls' + cls_cost + '.npy')
        return out_file, ap_matrix_out_file

    @staticmethod
    def parse_cost_list(costs, default_range=(4,20)):
        """Parse cost list."""
        if isinstance(costs, str):
            costs = costs.split(',')
        costs_list = [float(cost) for cost in costs]
        if default_range is not None:
            # default range as the start and the end
            for num in range(default_range[0], default_range[1]):
                costs_list.append(0.5**num)
        return costs_list

    @staticmethod
    def normalize_features(features):
        """numpy L2 normalization."""
        feats_norm = np.linalg.norm(features, axis=1)
        features = features / (feats_norm + 1e-5)[:, np.newaxis]
        return features

    @staticmethod
    def normalize_features_torch(features):
        """torch L2 normalization."""
        features = features / (torch.norm(features, p=2, dim=1, keepdim=True) + 1e-5)
        return features

    @staticmethod
    def load_input_data(data_file, targets_file):
        """Load the features and the targets."""
        targets = np.load(targets_file, encoding='latin1')
        features = np.array(np.load(data_file,
                                    encoding='latin1')).astype(np.float64)
        assert features.shape[0] == targets.shape[0], 'Mismatched #images'
        return features, targets

    @staticmethod
    def calculate_ap(rec, prec):
        """Computes the AP under the precision recall curve."""
        rec, prec = rec.reshape(rec.size, 1), prec.reshape(prec.size, 1)
        z, o = np.zeros((1, 1)), np.ones((1, 1))
        mrec, mpre = np.vstack((z, rec, o)), np.vstack((z, prec, z))
        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])

        indices = np.where(mrec[1:] != mrec[0:-1])[0] + 1
        ap = 0
        for i in indices:
            ap = ap + (mrec[i] - mrec[i - 1]) * mpre[i]
        return ap

    @staticmethod
    def get_precision_recall(targets, preds):
        """
        [P, R, score, ap] = get_precision_recall(targets, preds)
        Input    :
            targets  : number of occurrences of this class in the ith image
            preds    : score for this image.
        Output   :
            P, R   : precision and recall.
            score  : score which corresponds to the particular precision and recall.
            ap     : average precision.
        """
        # binarize targets
        targets = np.array(targets > 0, dtype=np.float32)
        tog = np.hstack((targets[:, np.newaxis].astype(np.float64),
                        preds[:, np.newaxis].astype(np.float64)))
        ind = np.argsort(preds)
        ind = ind[::-1]
        score = np.array([tog[i, 1] for i in ind])
        sortcounts = np.array([tog[i, 0] for i in ind])

        tp = sortcounts
        fp = sortcounts.copy()
        for i in range(sortcounts.shape[0]):
            if sortcounts[i] >= 1:
                fp[i] = 0.
            elif sortcounts[i] < 1:
                fp[i] = 1.
        P = np.cumsum(tp) / (np.cumsum(tp) + np.cumsum(fp))
        numinst = np.sum(targets)
        R = np.cumsum(tp) / numinst
        ap = SVMHelper.calculate_ap(R, P)
        return P, R, score, ap

    @staticmethod
    def get_low_shot_output_file(opts, cls, cost, suffix):
        """in case of low-shot training, we train for 5 independent samples
        (sample{}) and vary low-shot amount (k{}).

        The input data should have sample{}_k{} information that we extract in
        suffix below.
        """
        cls_cost = str(cls) + '_cost' + SVMHelper.py2_py3_compatible_cost(cost)
        out_file = osp.join(opts.output_path,
                            'cls' + cls_cost + '_' + suffix + '.pickle')
        return out_file

    @staticmethod
    def get_low_shot_svm_classes(targets, dataset="onehot"):
        """Get num_classes and cls_list information by dataset type."""
        # classes for which SVM testing should be done
        num_classes, cls_list = None, None
        if dataset == 'multi_label':  # VOC, COCO
            num_classes = targets.shape[1]
            cls_list = range(num_classes)
        elif dataset == 'onehot':
            # e.g., each image in places has a target cls [0, .... ,204] in Place205
            targets = targets.reshape(-1, 1)
            cls_list = list(set(targets[:, 0].tolist()))
            num_classes = len(cls_list)
        else:
            print_log('Dataset not recognized in SVM evaluation!')
        return num_classes, cls_list

    @staticmethod
    def get_cls_feats_labels(cls, features, targets, dataset="onehot"):
        """Get out_feats and out_cls_labels information by dataset type."""
        out_feats, out_cls_labels = None, None
        if dataset == 'multi_label':  # VOC, COCO
            cls_labels = targets[:, cls].astype(dtype=np.int32, copy=True)
            # find the indices for positive/negative imgs. Remove the ignore label.
            out_data_inds = (targets[:, cls] != -1)
            out_feats = features[out_data_inds]
            out_cls_labels = cls_labels[out_data_inds]
            # label 0 = not present, set it to -1 as svm train target.
            # Make the svm train target labels as -1, 1.
            out_cls_labels[np.where(out_cls_labels == 0)] = -1
        elif dataset == 'onehot':  # IN, Place205
            out_feats = features
            out_cls_labels = targets.astype(dtype=np.int32, copy=True)
            # for the given class, get the relevant positive/negative images and
            # make the label 1, -1
            cls_inds = np.where(targets[:, 0] == cls)
            non_cls_inds = (targets[:, 0] != cls)
            out_cls_labels[non_cls_inds] = -1
            out_cls_labels[cls_inds] = 1
            # finally reshape into the format taken by sklearn svm package.
            out_cls_labels = out_cls_labels.reshape(-1)
        else:
            raise Exception('args.dataset not recognized')
        return out_feats, out_cls_labels


def svm_task(cls, cost, features, targets, model_path):
    """The task function to train the model."""
    out_file, ap_out_file = SVMHelper.get_svm_train_output_files(
        cls, cost, model_path)
    clf = LinearSVC(
        C=cost,
        class_weight={
            1: 2,
            -1: 1
        },
        intercept_scaling=1.0,
        verbose=0,
        penalty='l2',
        loss='squared_hinge',
        tol=0.0001,
        dual=True,
        max_iter=2000,
    )
    cls_labels = targets[:, cls].astype(dtype=np.int32, copy=True)
    cls_labels[np.where(cls_labels == 0)] = -1
    ap_scores = cross_val_score(
        clf, features, cls_labels, cv=3, scoring='average_precision')
    clf.fit(features, cls_labels)
    np.save(ap_out_file, np.array([ap_scores.mean()]))
    with open(out_file, 'wb') as fwrite:
        pickle.dump(clf, fwrite)
    fwrite.close()
    return 0


def mp_helper(args):
    return svm_task(*args)


class LinearSVMClassifier():
    """Implements the Linear SVC classifier for evaluation.

    Args:
        dataset (str): Types of dataset in {'onehot', 'multi_label'} modes.
            Default to "onehot".
        costs_list (str): Costs for linear SVM seperateb by ','.
        default_cost (tuple, optional): Range of costs in (0,1] for SVM classifier.
            In multi-label classification scenarios (e.g., VOC), the default range
            is (4,20); in small-scale classification datasets, it can be removed.
            Default to (4,20).
        num_workers (int, optional): Number of parall workers to caculate
            SVM. When set to 0 or -1, only use a single process. Defaults to 0.
    """

    def __init__(self,
                 dataset="onehot",
                 costs_list="0.01,0.1,1",
                 default_cost=(4,20),
                 num_workers=0,
                ):
        super().__init__()
        self.dataset = dataset
        self.costs_list = costs_list
        self.default_cost = default_cost
        self.num_workers = num_workers
        self.model_path = None
        self.save_model = False
        assert default_cost is None or isinstance(default_cost, (list, tuple))

    @torch.no_grad()
    def evaluate(self,
                 train_features, train_targets, test_features, test_targets,
                 keyword, logger=None, topk=(1, 5)):
        """Computes linear SVM mAP.
        
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
        mmcv.mkdir_or_exist(self.model_path)
        
        # L2 normalize the features
        train_features = SVMHelper.normalize_features(train_features)
        test_features = SVMHelper.normalize_features(test_features)

        # parse the cost values for training the SVM on
        costs_list = SVMHelper.parse_cost_list(self.costs_list, self.default_cost)
        # classes for SVM training
        num_classes, cls_list = \
            SVMHelper.get_low_shot_svm_classes(train_targets, self.dataset)
        
        if self.dataset == "onehot" and len(train_targets.shape) == 1:
            train_targets = one_hot(
                torch.from_numpy(train_targets), num_classes).numpy()
            test_targets = one_hot(
                torch.from_numpy(test_targets), num_classes).numpy()

        # training
        if self.num_workers > 0:
            self.train_svm_parallel(
                train_features, train_targets, cls_list, costs_list)
        else:
            self.train_svm_unparallel(
                train_features, train_targets, cls_list, costs_list)
        
        # testing
        costs_list = self.get_chosen_costs(
            self.costs_list, num_classes, self.model_path)
        mAP = self.test_svm(
            test_features, test_targets, costs_list, num_classes)
        eval_res["{}_map".format(keyword)] = mAP
        if logger is not None and logger != 'silent':
            for k, v in eval_res.items():
                print_log("{}: {:.03f}".format(k, v), logger=logger)
        return eval_res

    def train_svm_parallel(self,
                           train_features, train_targets,
                           cls_list, costs_list):
        """Train SVM model parallel."""

        num_task = len(cls_list) * len(costs_list)
        args_cls, args_cost = list(), list()
        for cls in cls_list:
            for cost in costs_list:
                args_cls.append(cls)
                args_cost.append(cost)
        args_path = [self.model_path] * num_task
        args_data = [train_features] * num_task
        args_label = [train_targets] * num_task

        pool = mp.Pool(self.num_workers)
        for _ in tqdm(
                pool.imap_unordered(
                    mp_helper,
                    zip(args_cls, args_cost, args_data, args_label, args_path)),
                total=num_task):
            pass

    def train_svm_unparallel(self,
                             train_features, train_targets,
                             cls_list, costs_list):
        """Train SVM model with a single process."""
        for cls_idx in tqdm(range(len(cls_list))):
            cls = cls_list[cls_idx]
            for cost_idx in range(len(costs_list)):
                cost = costs_list[cost_idx]
                svm_task(cls, cost, train_features, train_targets, self.model_path)

    def get_chosen_costs(self, costs_list, num_classes, model_path):
        """get the chosen cost that maximizes the cross-validation AP per class."""
        costs_list = SVMHelper.parse_cost_list(costs_list, self.default_cost)
        train_ap_matrix = np.zeros((num_classes, len(costs_list)))
        for cls in range(num_classes):
            for cost_idx in range(len(costs_list)):
                cost = costs_list[cost_idx]
                _, ap_out_file = SVMHelper.get_svm_train_output_files(
                    cls, cost, model_path)
                train_ap_matrix[cls][cost_idx] = float(
                    np.load(ap_out_file, encoding='latin1')[0])
        argmax_cls = np.argmax(train_ap_matrix, axis=1)
        chosen_cost = [costs_list[idx] for idx in argmax_cls]
        np.save(
            osp.join(model_path, 'crossval_ap.npy'),
            np.array(train_ap_matrix))
        np.save(
            osp.join(model_path, 'chosen_cost.npy'), np.array(chosen_cost))
        return np.array(chosen_cost)

    def test_svm(self,
                 test_features, test_targets, costs_list, num_classes):
        """Test SVM model."""
        ap_matrix = np.zeros((num_classes, 1))
        for cls in range(num_classes):
            cost = costs_list[cls]
            model_file = osp.join(
                self.model_path, f"cls{str(cls)}_cost{str(cost)}.pickle")
            with open(model_file, 'rb') as fopen:
                if six.PY2:
                    model = pickle.load(fopen)
                else:
                    model = pickle.load(fopen, encoding='latin1')
            prediction = model.decision_function(test_features)
            
            cls_labels = test_targets[:, cls]
            # meaning of labels in VOC/COCO original loaded target files:
            # label 0 = not present, set it to -1 as svm train target
            # label 1 = present. Make the svm train target labels as -1, 1.
            evaluate_data_inds = (test_targets[:, cls] != -1)
            eval_preds = prediction[evaluate_data_inds]
            eval_cls_labels = cls_labels[evaluate_data_inds]
            eval_cls_labels[np.where(eval_cls_labels == 0)] = -1
            _, _, _, ap = SVMHelper.get_precision_recall(
                eval_cls_labels, eval_preds)
            ap_matrix[cls][0] = ap * 100  # as percent %

        mAP = np.mean(ap_matrix)
        if self.save_model:
            np.save(osp.join(
                self.model_path, 'test_svm_ap.npy'), np.array(ap_matrix))
        return mAP
