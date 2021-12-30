# my implementation of visualization
import argparse
import importlib
import os
import time

import mmcv
import torch
import torchvision
from mmcv.parallel import MMDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

from openmixup.datasets import build_dataloader, build_dataset
from openmixup.models import build_model
from openmixup.utils import (get_root_logger, dist_forward_collect,
                               nondist_forward_collect, traverse_replace)

import numpy as np
# add your visualization tools
import umap
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def single_gpu_test(model, data_loader):
    print("running signal gpu test...")
    model.eval()
    func = lambda **x: model(mode='test', **x)
    results = nondist_forward_collect(func, data_loader,
                                      len(data_loader.dataset))
    return results


def parse_args(ckpt_name, work_dir,
            cnn_arch="r50", data_name="imagenet", tool_name="umap", binary=False, **kwargs):
    assert cnn_arch in ["r50", "r18", "lenet"]
    config = "configs/benchmarks/linear_classification/{}/{}_rep_{}.py".format(data_name, cnn_arch, data_name)

    args = {
        "data_name": data_name,
        "binary": binary,  # add for Pets-37 dataset
        "config": config,
        "checkpoint": ckpt_name,
        "work_dir": work_dir,
        "tool_name": tool_name,
        "launcher": None,  # 'none', 'pytorch', 'slurm', 'mpi'
        "local_rank": 0,
        "port": 29590,
    }
    return args


def load_imagenet_labels(root=None):
    # only for class=10 ImageNet, unshuffle targets
    label = [i // 1300 for i in range(13000)]  # 10 class
    label = np.array(label, dtype=np.int32)
    print("Notice: only visualize 10 class selected in ImageNet-1k!")
    return label


def load_tiny_imagenet_labels(root=None):
    label = list()
    if root is None:
        tiny_target_path = './data/TinyImagenet200/meta/train_20class_labeled.txt'
    else:
        tiny_target_path = root
    print("Notice: data root={}.".format(tiny_target_path))
    with open(tiny_target_path, "r") as fp:
        lines = fp.readlines()
        for l in lines:
            label.append(l.split("\n")[0].split(" ")[1])
    fp.close()
    label = np.array(label, dtype=np.int32)
    return label


def load_cifar10_labels(root=None):
    label = list()
    if root is None:
        root = "./data/cifar10/"
    print("Notice: cifar10 data root={}, using unshuffle labels.".format(root))
    cifar = torchvision.datasets.CIFAR10(
        root=root, train=False, download=False)
    label = np.array(cifar.targets, dtype=np.int32)
    return label


def load_cifar100_labels(root=None):
    label = list()
    if root is None:
        root = "./data/cifar100/"
    print("Notice: cifar100 data root={}, using unshuffle labels.".format(root))
    cifar = torchvision.datasets.CIFAR100(
        root=root, train=False, download=False)
    label = np.array(cifar.targets, dtype=np.int32)
    return label


def load_mnist_labels(root=None):
    label = list()
    if root is None:
        root = "./data/"
    print("Notice: MNIST data root={}, using unshuffle labels.".format(root))
    mnist = torchvision.datasets.MNIST(
        root=root, train=False, download=False)
    label = np.array(mnist.targets, dtype=np.int32)
    return label


def load_fmnist_labels(root=None):
    label = list()
    if root is None:
        root = "./data/"
    print("Notice: FMNIST data root={}, using unshuffle labels.".format(root))
    mnist = torchvision.datasets.FashionMNIST(
        root=root, train=False, download=False)
    label = np.array(mnist.targets, dtype=np.int32)
    return label


def load_stl10_labels(root=None):
    label = list()
    if root is None:
        root = './data/stl10_binary/test_y.bin'
    print("Notice: STL10 data root={}, using unshuffle labels.".format(root))
    try:
        with open(root, "r") as fp:
            lines = fp.readlines()
            for l in lines:
                label.append(l.split("\n")[0].split(" ")[1])
        fp.close()
        label = np.array(label, dtype=np.int32)
    except:
        from tools.stl10_official_util import read_labels
        label = np.array(read_labels(root), dtype=np.int32)  # same for cifar-10, stl-10
    return label


def load_pets_labels(root=None, binary=False):
    label = list()
    if root is None:
        pets_target_path = "/usr/commondata/public/Pets37/classification_meta_0/test_labeled.txt"
    else:
        pets_target_path = root
    print("Notice: Pets data root={}.".format(pets_target_path))
    with open(pets_target_path, "r") as fp:
        lines = fp.readlines()
        for l in lines:
            if binary:
                label.append( int(l[0].islower()) )
            else:
                label.append(l.split("\n")[0].split(" ")[1])
    fp.close()
    label = np.array(label, dtype=np.int32)
    return label


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def main_runner(args):

    cfg = mmcv.Config.fromfile(args["config"])
    # set cudnn_benchmark
    torch.backends.cudnn.benchmark = True
    # update configs according to CLI args
    if args["work_dir"] is not None:
        cfg.work_dir = args["work_dir"]
    cfg.model.pretrained = None  # ensure to use checkpoint rather than pretraining

    # check memcached package exists
    if importlib.util.find_spec('mc') is None:
        traverse_replace(cfg, 'memcached', False)
    
    distributed = False
    # build the dataloader
    dataset = build_dataset(cfg.data.val)
    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=cfg.data.imgs_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)
    
    # build the model and load checkpoint
    model = build_model(cfg.model)
    
    save_name = "None"
    embed_npy_path = []

    if args["checkpoint"] is not None:
        checkpoint = torch.load(args["checkpoint"])
        state_dict = checkpoint['state_dict']
        save_name = args["checkpoint"].split('/')[-1].split('.')[0]
        # choose param for this Model
        model_dict = model.state_dict()
        pretrained = {"backbone."+k: v for k, v in state_dict.items() if "backbone."+k in model_dict}
        print('load pretrained model={}, model key_len={}, pretrains key_len={}'.format(
            args["checkpoint"], len(list(model_dict.keys())), len(list(pretrained.keys())) ))
        model_dict.update(pretrained)
        model.load_state_dict(model_dict)

    # build DDP model    
    model = MMDataParallel(model, device_ids=[0])
    outputs = single_gpu_test(model, data_loader)

    # save encode representation as .npy
    for k in outputs.keys():
        print("output key={}, shape={}".format(k, outputs[k].shape))
        embed_npy_path.append("{}_{}".format(k, save_name))
        save_output = outputs[k]  # outputs[k].cpu().numpy()
        save_output = np.reshape(save_output, (save_output.shape[0], -1))
        print("saving output={}".format(save_output.shape))
        mkdirs("work_dirs/representations/")
        np.save("work_dirs/representations/{}_{}.npy".format(k, save_name), save_output)
    
    # run UMAP to visualiza embed to dim=2
    for npy_path in embed_npy_path:
        embed_path = "work_dirs/representations/{}_{}.npy".format(npy_path, args["tool_name"])
        mkdirs("work_dirs/plot/{}".format(args["data_name"]))
        try:
            plot_path = "work_dirs/plot/{}/{}_{}e.png".format(args["data_name"], npy_path, str(args["epoch"]))
        except:
            plot_path = "work_dirs/plot/{}/{}.png".format(args["data_name"], npy_path)
            
        # loading dataset labels
        try:
            kwargs = "root=None"
            if args.get("binary", False):
                assert args["data_name"] == "pets"
                kwargs = "root=None, binary=True"
            label = eval("load_{}_labels({})".format(args["data_name"], kwargs))
        except:
            print("Please add 'load_{}_labels()' to load labels!".format(args["data_name"]))
            exit()

        _embed_npy = np.load("work_dirs/representations/{}.npy".format(npy_path))
        assert label.shape[0] == _embed_npy.shape[0] and _embed_npy.shape[1] >= 2
        # choose visualization tools
        if args["tool_name"] == "umap":
            _embed_2d = umap.UMAP(n_components=2).fit_transform(_embed_npy)
        elif args["tool_name"] == "pca":
            pca = PCA(n_components=2)
            _embed_2d = pca.fit_transform(_embed_npy)
        else:
            print("Please choose a valid visualization tool!")
            exit()
        np.save(embed_path, _embed_2d)
        print("loaded embedding of {}: {}".format(args["tool_name"], _embed_2d.shape))
        
        # plot with embedding dim=2
        plt.figure(figsize=(8, 8))
        plt.scatter(
            _embed_2d[:, 0], _embed_2d[:, 1], c=label, s=1,
            cmap='rainbow'
        )
        plt.colorbar()
        plt.title(npy_path)
        plt.savefig(plot_path, dpi=300)
        print("finish ploting: {} --> {}".format(npy_path, plot_path))


def extract_weights(checkpoint, save_path=None, save_name=None):
    assert save_name is not None
    mkdirs(save_path)

    ck = torch.load(checkpoint, map_location=torch.device('cpu'))
    output_dict = dict(state_dict=dict(), author="openmixup+v1.0_CAIRI")
    has_backbone = False
    for key, value in ck['state_dict'].items():
        if key.startswith('backbone'):
            output_dict['state_dict'][key[9:]] = value
            has_backbone = True
    if not has_backbone:
        raise Exception("Cannot find a backbone module in the checkpoint.")
    
    save_path = os.path.join(save_path, save_name)
    torch.save(output_dict, save_path)
    print("successfually save model to {}".format(save_path))


def main():
    """ Usage of visualization
        version 01.05
    
    Install:
        pip install umap-learn
    
    Settings:
        cnn_arch: Only support ResNet ["r18", "r50"] and LeNet ["lenet"]
        data_name: Choose a dataset, ["imagenet", "tiny_imagenet" "stl10", "cifar10", "cifar100", "pets", "mnist"]
        dirs_name: Support ["selfsup", "classification"], i.e. "./configs/selfsup" and "./configs/classification"
        binary_class_pets: Only for Pets, "dogs", "cats"
        model_dict: Model_name and config file_name as a key-value pair
        epoch_num: A list of epoch nums
        tool_name: Choose a visualization tool, ["umap", "pca", "tsne"]
    
    Running:
        1. set params in "basic_dict".
        2. python tools/visualize_embedding.py
        3. find plot results in ./work_dirs/plot/
    """

    data_choice = ["imagenet", "tiny_imagenet", "stl10", "cifar10", "cifar100", "pets", "mnist", "fmnist"]
    mode_choice = ["selfsup", "classification"]
    basic_dict = dict(
        cnn_arch="r50",  # ["r50", "r18"]
        # cnn_arch="r18",
        # cnn_arch="lenet",
        # ----------------------------------------------------------------
        data_name="stl10",
        # data_name="cifar10",
        # data_name="cifar100",
        # data_name="mnist",
        # data_name="fmnist",
        # ----------------------------------------------------------------
        dirs_name="selfsup",
        # dirs_name="classification",
        # ----------------------------------------------------------------
        tool_name="umap",
        binary=False,  # [True, False], only for Pets dataset
    )
    assert basic_dict["data_name"] in data_choice
    assert basic_dict["dirs_name"] in mode_choice

    # set the epoch number your want to visualize
    epoch_nums = [100, 200]
    # epoch_nums = [i*100 for i in range(1, 6)]
    epoch_nums = [i*100 for i in range(1, 9)]

    # Optional: add inner_folder when your config path likes
    #    "configs/classification/tiny_imagenet/baseline/r18_cosine_bs100_ep200.py"
    # inner_folder = "mixup"
    inner_folder = None

    model_dict = {
        # ------------------------ supervised visualization ----------------------------------
        # --> example 1
        # "data name":    "your cls config name",
        # "cifar10":      "r18_cifar_mixup_cosine_bs100_ep200",   # set inner_folder="mixup"
        # --------------------- self-supervised visualization --------------------------------
        # --> example 2
        # "model name":   "your SSL config name",
        "moco":         "r50_v2_10percent_1204_bs256_ep400",    # set inner_folder=None
    }

    for key,ckpt in model_dict.items():
        if ckpt.find("/") == -1:  # normal
            if isinstance(epoch_nums, float):
                epoch_nums = [epoch_nums]
            assert isinstance(epoch_nums, list)
            # synthesize ckpt path
            if inner_folder is None:
                base_path = "work_dirs/{}/{}/{}".format(basic_dict["dirs_name"], key, ckpt)
            else:
                base_path = "work_dirs/{}/{}/{}/{}".format(basic_dict["dirs_name"], key, inner_folder, ckpt)
            for epoch_num in epoch_nums:
                extract_weights(
                    checkpoint="{}/epoch_{}.pth".format(base_path, epoch_num),
                    save_path="work_dirs/my_pretrains/{}/".format(key),
                    save_name="{}_{}.pth".format(key, ckpt))
                print("Start running: {} -> {}...".format(key, ckpt))
                args = parse_args(
                    "work_dirs/my_pretrains/{}/{}_{}.pth".format(key, key, ckpt),
                    "./work_dirs", **basic_dict
                )
                args["epoch"] = epoch_num
                # run test and plot
                main_runner(args)
        else:
            name = "{}_{}".format(ckpt.split("/")[-3], ckpt.split("/")[-2])
            extract_weights(
                checkpoint=ckpt,
                save_path="work_dirs/my_pretrains/{}/".format(key),
                save_name="{}.pth".format(name))
            print("Start running: {}...".format(ckpt))
            args = parse_args(
                "work_dirs/my_pretrains/{}/{}.pth".format(key, name),
                "./work_dirs",  **basic_dict
            )
            # run test and plot
            main_runner(args)


if __name__ == '__main__':
    main()

    # kill python: ps aux|grep python|grep -v grep|cut -c 9-15|xargs kill -9