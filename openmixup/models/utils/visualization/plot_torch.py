import os
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt


class PlotTensor:
    """Plot torch tensor as matplotlib figure.

    Args:
        apply_inv (bool): Whether to apply inverse normalization.
    """

    def __init__(self, apply_inv=True) -> None:
        trans_cifar = [
            torchvision.transforms.Normalize(
                mean=[ 0., 0., 0. ], std=[1 / 0.2023, 1 / 0.1994, 1 / 0.201]),
            torchvision.transforms.Normalize(
                mean=[-0.4914, -0.4822, -0.4465], std=[ 1., 1., 1. ])]
        trans_in = [
            torchvision.transforms.Normalize(
                mean=[ 0., 0., 0. ], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
            torchvision.transforms.Normalize(
                mean=[-0.485, -0.456, -0.406], std=[ 1., 1., 1. ])]
        if apply_inv:
            self.invTrans_cifar = torchvision.transforms.Compose(trans_cifar)
            self.invTrans_in = torchvision.transforms.Compose(trans_in)

    def plot(self,
             img, nrow=4, title_name=None, save_name=None,
             dpi=None, cmap='gray', apply_inv=True, overwrite=False):
        assert save_name is not None
        assert img.size(0) % nrow == 0
        ncol = img.size(0) // nrow
        if ncol > nrow:
            ncol = nrow
            nrow = img.size(0) // ncol
        img_grid = torchvision.utils.make_grid(img, nrow=nrow, pad_value=0)

        if img.size(1) == 1:
            cmap = getattr(plt.cm, cmap, plt.cm.jet)
        else:
            cmap = None
        if apply_inv:
            if img.size(2) <= 64:
                img_grid = self.invTrans_cifar(img_grid)
            else:
                img_grid = self.invTrans_in(img_grid)
        img_grid = torch.clip(img_grid * 255, 0, 255).int()
        img_grid = np.transpose(img_grid.detach().cpu().numpy(), (1, 2, 0))
        fig = plt.figure(figsize=(nrow * 2, ncol * 2))
        plt.imshow(img_grid, cmap=cmap)
        if title_name is not None:
            plt.title(title_name)
        if not os.path.exists(save_name) or overwrite:
            plt.savefig(save_name, dpi=dpi)
        plt.close()
