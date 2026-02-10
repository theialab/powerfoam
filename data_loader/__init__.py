import os

import numpy as np
import torch

from .colmap import COLMAPDataset
from .blender import BlenderDataset
from .scannetpp import ScannetPPDataset


dataset_dict = {
    "colmap": COLMAPDataset,
    "blender": BlenderDataset,
    "scannetpp": ScannetPPDataset,
}


def get_up(c2ws):
    right = c2ws[:, :3, 0]
    down = c2ws[:, :3, 1]
    forward = c2ws[:, :3, 2]

    A = torch.einsum("bi,bj->bij", right, right).sum(dim=0)
    A += torch.einsum("bi,bj->bij", forward, forward).sum(dim=0) * 0.02

    l, V = torch.linalg.eig(A)

    min_idx = torch.argmin(l.real)
    global_up = V[:, min_idx].real
    global_up *= torch.einsum("bi,i->b", -down, global_up).sum().sign()

    return global_up


class DataHandler:
    def __init__(self, dataset_args):
        self.args = dataset_args
        self.img_wh = None

    def reload(self, split, downsample):
        self.split = split
        data_dir = os.path.join(self.args.data_path, self.args.scene)
        dataset = dataset_dict[self.args.dataset]
        if downsample is not None:
            split_dataset = dataset(
                data_dir,
                split=split,
                downsample=downsample,
                alpha_format_on_disk=self.args.alpha_format_on_disk,
                use_metric3d=self.args.use_metric3d,
            )
        else:
            split_dataset = dataset(
                data_dir,
                split=split,
                alpha_format_on_disk=self.args.alpha_format_on_disk,
                use_metric3d=self.args.use_metric3d,
            )
        self.img_wh = split_dataset.img_wh
        self.c2ws = split_dataset.poses
        self.cameras = split_dataset.all_cameras
        self.rgbs = split_dataset.all_rgbs.pin_memory()
        self.alphas = split_dataset.all_alphas.pin_memory()
        if self.args.use_metric3d:
            self.normals = split_dataset.all_normals.pin_memory()
        else:
            self.normals = None

        self.viewer_up = get_up(self.c2ws)
        self.viewer_pos = self.c2ws[0, :3, 3]
        self.viewer_forward = self.c2ws[0, :3, 2]

        self.points3D = split_dataset.points3D
        self.points3D_colors = split_dataset.points3D_color

    def get_iter(self):
        stream = torch.cuda.Stream()
        next_batch = None

        def start_fetch(i):
            nonlocal next_batch
            with torch.cuda.stream(stream):
                camera = self.cameras[i]
                rgb = self.rgbs[i].cuda(non_blocking=True)
                alpha = self.alphas[i].cuda(non_blocking=True)
                if self.args.use_metric3d:
                    normal = self.normals[i].cuda(non_blocking=True)
                else:
                    normal = None
                next_batch = (camera, rgb, alpha, normal)

        start_fetch(0)

        while True:
            if self.split == "train":
                indices = np.random.permutation(len(self.rgbs))
            else:
                indices = np.arange(len(self.rgbs))

            for i in indices:
                torch.cuda.current_stream().wait_stream(stream)
                current_batch = next_batch
                start_fetch(i)

                yield current_batch
