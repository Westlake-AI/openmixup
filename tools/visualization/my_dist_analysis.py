import os
import numpy as np
import random
from tqdm import tqdm
from PIL import Image
from PIL import ImageFilter

import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F

from matplotlib import pyplot as plt


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def Distance_squared(x, y, min_eye=1e-12):
    """ L2 dist """
    x = x.reshape(x.shape[0], -1)
    y = y.reshape(y.shape[0], -1)

    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())  # out = 1 * dist + (-2) * x@y.t()
    return dist


def hist_numpy(x_list, bin_num=200, save_name="none"):
    """ get histogram of a numpy array X """
    hist_list, bin_list = list(), list()
    rects = list()
    color_list = ["blue", "red", "green", "yellow", "black"]
    assert len(x_list) <= len(color_list)
    
    nums = range(bin_num)
    
    for i,x in enumerate(x_list):
        print("plot {}: input shape={}".format(i, x.shape))
        x_min = np.min(x)
        x_max = np.max(x)
        x = (x - x_min) / (x_max - x_min) * bin_num
        # hist = {i:0 for i in range(bin_num)}
        # print(hist)
        hist1, bins = np.histogram(x, bin_num, [0, bin_num-1])
        # cdf = hist1.cumsum()
        
        # save bar img
        rect = plt.bar(nums, height=hist1, width=1.0, alpha=1.0 / len(x_list), color=color_list[i], label=save_name+"_"+str(i))
        rects.append(rect)
    
    # save bar img
    # plt.bar(nums, height=hist1, width=0.4, color="blue", label=save_name)
    plt.savefig("./work_dirs/plot/{}_hist.png".format(save_name), dpi=100)
    plt.close()
    # save hist to .npy
    # np.save("./plot/{}.npy".format(save_name), hist1)


def load_dataset_with_transform(aug_transform=None):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize  # not use 1019
    ])

    trainset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=None
    )

    data, label = trainset.data, trainset.targets

    # apply basic transformation
    data_list = []
    for i in tqdm(range(data.shape[0])):
        _data = transform(data[i, :, :, :])
        _data = _data.unsqueeze(0)
        data_list.append(_data)
    # reshape and return
    data_ori = torch.cat(data_list, dim=0)
    data_ori = data_ori.reshape(data_ori.shape[0], -1)
    # print(type(data_ori))

    if aug_transform is not None:
        data_aug = []
        for i in tqdm(range(data.shape[0])):
            # _data = Image.fromarray(data[i, :, :, :].astype('uint8')).convert('RGB')
            _data = Image.fromarray(data[i, :, :, :]).convert('RGB')
            _data = aug_transform(_data)
            _data = _data.unsqueeze(0)
            data_aug.append(_data)
            
        # reshape and return
        data_aug = torch.cat(data_aug, dim=0)
        # print(type(data_aug))
        data_aug = data_aug.reshape(data_aug.shape[0], -1)
        return data_ori, data_aug, label
    
    return data_ori, label


def load_dataset_class10():
    """ ImageNet: class 10 dataset """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize(size=96),
        transforms.CenterCrop(size=96),
        transforms.ToTensor(),
        normalize  # not use 1019
    ])

    path = "/imagenet/meta/train_labeled_10class_0123_8081_154155_404_407.txt"
    pathImage = "/imagenet/train/"
    
    path_txt = open(path, "r").read().split("\n")
    data = []
    datastr = []
    targets = []
    for txt in tqdm(path_txt):
        if len(txt) > 4:
            pathI = txt.split(" ")[0]
            datastr.append(pathImage + pathI)
            targets.append(int(txt.split(" ")[1]))
            # read image with PIL
            image = Image.open(pathImage + pathI).convert("RGB")
            image.resize((96, 96))
            data.append(transform(image).unsqueeze(0))
    
    data = torch.cat(data, dim=0)
    data = data.reshape(data.shape[0], -1)
    # label = torch.cat(targets, dim=0)
    label = targets
    print("loaded imagenet class10 with data={}, label={}".format(data.shape, len(label)))
    return data, label


def load_embedding_class10(file_path=None):
    """ load .npy results of class10 """
    # only for class=10 ImageNet
    label = [i // 1300 for i in range(13000)]  # 10 class
    label = np.array(label, dtype=np.int32)
    embed = np.load(file_path)
    return embed, label



# cosine dist
def dist_basic(data, label=None, mode="cosine"):
    """ basic version """
    bs = data.size(0)
    data = data.reshape(bs, -1)
    if mode == "cosine":
        data_norm = F.normalize(data, dim=1, p=2)
        data_dist = data_norm.mm(data_norm.t())
        dist = (1.0 - data_dist) / 1.0
    elif mode == "L2":
        dist = Distance_squared(data, data)

    return dist.numpy()


def dist_binary(data, label=None, mode="cosine"):
    """ compute positive and negative samples dist """
    # index = 50000 # 1000
    bs = data.size(0)
    data = data.reshape(bs, -1)

    # data = data[:index, :]
    # label = torch.tensor(label[:index]).unsqueeze(0)
    label = torch.tensor(label).unsqueeze(0)
    pos_mask = label == label.t()
    neg_mask = label != label.t()
    print("mask shape={}".format(pos_mask.shape))

    if mode == "cosine":
        data_norm = F.normalize(data, dim=1, p=2)
        data_dist = data_norm.mm(data_norm.t())
        dist = (1.0 - data_dist) / 1.0
    elif mode == "L2":
        dist = Distance_squared(data, data)

    dist_pos = torch.masked_select(dist, pos_mask)
    dist_neg = torch.masked_select(dist, neg_mask)
    print("sum={}; pos dist num={}, neg dist num={}".format(
        dist_pos.shape[0]+dist_neg.shape[0], dist_pos.shape[0], dist_neg.shape[0]))

    return dist_pos.numpy(), dist_neg.numpy()


def dist_augmentation(data, data_aug=None, label=None, mode="cosine"):
    """ using augmentation: positive and negative samples dist """
    # index = 50000 # 1000
    bs = data.size(0)
    if type(data) != type(data_aug):
        data = torch.tensor(data)
        data_aug = torch.tensor(data_aug)
    data = data.reshape(bs, -1)
    data_aug = data_aug.reshape(bs, -1)

    # data = data[:index, :]
    # label = torch.tensor(label[:index]).unsqueeze(0)
    label = torch.tensor(label).unsqueeze(0)
    pos_mask = label == label.t()
    neg_mask = label != label.t()
    print("mask shape={}".format(pos_mask.shape))

    if mode == "cosine":
        # basic
        data_norm = F.normalize(data, dim=1, p=2)
        data_dist = data_norm.mm(data_norm.t())
        dist = (1.0 - data_dist) / 1.0
        # augmentation
        data_aug_norm = F.normalize(data_aug, dim=1, p=2)
        data_aug_dist = data_aug_norm.mm(data_norm.t())
        dist_aug = (1.0 - data_aug_dist) / 1.0
    elif mode == "L2":
        # basic
        dist = Distance_squared(data, data)
        # augmentation
        dist_aug = Distance_squared(data, data_aug)

    dist_pos = torch.masked_select(dist, pos_mask)
    dist_neg = torch.masked_select(dist, neg_mask)
    dist_aug_pos = torch.masked_select(dist_aug, pos_mask)
    # print("sum={}; pos dist num={}, neg dist num={}".format(
    #     dist_pos.shape[0]+dist_neg.shape[0], dist_pos.shape[0], dist_neg.shape[0]))

    return dist_pos.numpy(), dist_neg.numpy(), dist_aug_pos.numpy()



# data_name = "cifar"
data_name = "imagenet"

if data_name == "cifar":

    # (1) basic mode
    # data, label = load_dataset_with_transform()  # Cifar-10
    # bs = data.size(0)
    # dist = dist_basic(data, label, "cosine")
    # print("cosine dist: shape={}, min_max=[{}, {}], mean={}, std={}".format(
    #     dist.shape, np.max(dist), np.min(dist), np.mean(dist), np.std(dist)))
    # hist_numpy([dist], bin_num=400, save_name="cifar10_cosine_basic")

    # dist = dist_basic(data, label, "L2")
    # print("L2 dist: shape={}, min_max=[{}, {}], mean={}, std={}".format(
    #     dist.shape, np.max(dist), np.min(dist), np.mean(dist), np.std(dist)))
    # hist_numpy([dist], bin_num=400, save_name="cifar10_L2_basic")

    # (2) binary mode
    # data, label = load_dataset_with_transform()  # Cifar-10
    # bs = data.size(0)
    # dist_pos, dist_neg = dist_binary(data, label, "cosine")
    # hist_numpy([dist_pos, dist_neg], bin_num=200, save_name="cifar10_cosine_binary")
    # dist_pos, dist_neg = dist_binary(data, label, "L2")
    # hist_numpy([dist_pos, dist_neg], bin_num=200, save_name="cifar10_L2_binary")

    # (3) augmentation mode
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_aug = transforms.Compose([
        # transforms.Resize(32),
        # transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
        transforms.RandomResizedCrop(32, scale=(0.5, 1.0)),
        transforms.RandomApply(
            [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8  # not strengthened
        ),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    data, data_aug, label = load_dataset_with_transform(aug_transform=transform_aug)  # Cifar-10
    # print(type(data), type(data_aug), type(label))
    bs = data.size(0)
    # mode = "cosine"
    mode = "L2"
    dist_pos, dist_neg, dist_aug_pos = dist_augmentation(data, data_aug, label, mode)
    hist_numpy([dist_pos, dist_neg, dist_aug_pos], bin_num=200, save_name="cifar10_{}_aug_binary".format(mode))

elif data_name == "imagenet":
    # (2) binary mode
    # data, label = load_dataset_class10()  # Imagenet class 10
    # bs = data.size(0)
    # dist_pos, dist_neg = dist_binary(data, label, "cosine")
    # hist_numpy([dist_pos, dist_neg], bin_num=200, save_name="imagenet_class10_cosine_binary")
    # dist_pos, dist_neg = dist_binary(data, label, "L2")
    # hist_numpy([dist_pos, dist_neg], bin_num=200, save_name="imagenet_class10_L2_binary")

    # (3) ImageNet class 10: raw visualization
    mode = "cosine"
    # mode = "L2"
    file_base = "/home/data/185-backup/home/lsy/MLDLv2_1030_download/"
    # file_path = "/xihu/lsy/MLDLv2-dev/work_dirs/representations/baseline/imagenet_supervised/head0_imagenet_r50_8gpu.npy"
    # file_path = "/xihu/lsy/openselfsup_0827/work_dirs/representations/simclr_r50_bs1024_ep200/epoch_200.npy"
    # file_path = file_base + "work_dirs/representations/head0_r50_1109_class10_cosineXZ_PM4_4gpu.npy"
    file_path = file_base + "work_dirs/representations/head0_r50_1109_class10_cosine_PM1_4gpu_repeat.npy"
    
    # --> ImageNet class 10: embedding visualization
    embed, label = load_embedding_class10(file_path)

    # add log
    # embed = np.log(1 + embed)
    # add exp
    # embed = np.exp(embed - 1.)  # OK
    # embed = np.power(embed - 1., 2)  # Error!!!
    embed = np.power(1.8, embed - 2.5)  #

    dist_pos, dist_neg = dist_binary(torch.tensor(embed), torch.tensor(label), mode)
    model_name = file_path.split("/")[-2]
    print("saving model name={}".format(model_name))
    hist_numpy([dist_pos, dist_neg], bin_num=800, save_name="class10_{}_{}_binary".format(mode, model_name))
