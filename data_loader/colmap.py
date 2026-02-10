import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn.functional as F
import pycolmap

from powerfoam.camera import TorchCamera
from .metric3D import Metric3DEstimator


def image_plane_basis(camera):
    pix_min = np.array([0.5, 0.5], dtype=np.float32)
    ip_min = camera.cam_from_img(pix_min)
    right = np.array([-ip_min[0], 0.0, 0.0], dtype=np.float32)
    up = np.array([0.0, ip_min[1], 0.0], dtype=np.float32)
    return torch.from_numpy(right), torch.from_numpy(up)


class COLMAPDataset:
    def __init__(self, datadir, split, downsample, alpha_format_on_disk, use_metric3d):
        assert downsample in [1, 2, 4, 8]

        self.root_dir = datadir
        self.colmap_dir = os.path.join(datadir, "sparse/0/")
        self.split = split
        self.downsample = downsample
        self.alpha_format_on_disk = alpha_format_on_disk
        self.use_metric3d = use_metric3d

        if downsample == 1:
            images_dir = os.path.join(datadir, "images")
        else:
            images_dir = os.path.join(datadir, f"images_{downsample}")

        if not os.path.exists(images_dir):
            raise ValueError(f"Images directory {images_dir} not found")

        self.reconstruction = pycolmap.Reconstruction()
        self.reconstruction.read(self.colmap_dir)

        if len(self.reconstruction.cameras) > 1:
            raise ValueError("Multiple cameras are not supported")

        names = sorted(im.name for im in self.reconstruction.images.values())
        indices = np.arange(len(names))

        metric3d_dir = os.path.join(datadir, "metric3d")
        if use_metric3d and not os.path.exists(metric3d_dir):
            print("Precomputed Metric3D data not found; running now...")
            os.makedirs(metric3d_dir)
            input_dir = os.path.join(datadir, "images")
            input_paths = list(os.path.join(input_dir, name) for name in names)
            estimator = Metric3DEstimator()
            estimator.process_dir(input_paths, metric3d_dir)

        if split == "train":
            names = list(np.array(names)[indices % 8 != 0])
        elif split == "test":
            names = list(np.array(names)[indices % 8 == 0])
        else:
            raise ValueError(f"Invalid split: {split}")

        names = list(str(name) for name in names)

        im = Image.open(os.path.join(images_dir, names[0]))
        self.img_wh = im.size
        im.close()

        self.camera = list(self.reconstruction.cameras.values())[0]
        self.camera.rescale(self.img_wh[0], self.img_wh[1])

        cam_space_right, cam_space_up = image_plane_basis(self.camera)

        self.images = []
        for name in names:
            image = None
            for image_id in self.reconstruction.images:
                image = self.reconstruction.images[image_id]
                if image.name == name:
                    break

            if image is None:
                raise ValueError(f"Image {name} not found in COLMAP reconstruction")

            self.images.append(image)

        self.poses = []
        self.all_cameras = []
        self.all_rgbs = []
        self.all_alphas = []
        for image in tqdm(self.images):
            c2w = torch.tensor(
                image.cam_from_world().inverse().matrix(), dtype=torch.float32
            )
            self.poses.append(c2w)
            world_right = torch.einsum("j,kj->k", cam_space_right, c2w[:, :3])
            world_up = torch.einsum("j,kj->k", cam_space_up, c2w[:, :3])

            self.all_cameras.append(
                TorchCamera(
                    eye=c2w[:3, 3].pin_memory(),
                    right=world_right.pin_memory(),
                    up=world_up.pin_memory(),
                    width=self.img_wh[0],
                    height=self.img_wh[1],
                )
            )

            im = Image.open(os.path.join(images_dir, image.name))
            rgba = np.array(im.convert("RGBA"), dtype=np.float32) / 255.0
            if self.alpha_format_on_disk == "premultiplied":
                alphas = torch.tensor(rgba[..., 3], dtype=torch.float32)
                rgbs = torch.tensor(rgba[..., :3], dtype=torch.float32)
            elif self.alpha_format_on_disk == "straight":
                alphas = torch.tensor(rgba[..., 3], dtype=torch.float32)
                rgbs = torch.tensor(rgba[..., :3] * rgba[..., 3:4], dtype=torch.float32)
            else:
                raise ValueError(
                    f"Unsupported alpha format on disk: {self.alpha_format_on_disk}"
                )
            im.close()

            self.all_rgbs.append(rgbs)
            self.all_alphas.append(alphas)

        self.poses = torch.stack(self.poses)
        self.all_rgbs = torch.stack(self.all_rgbs)
        self.all_alphas = torch.stack(self.all_alphas)

        self.points3D = []
        self.points3D_color = []
        for point in self.reconstruction.points3D.values():
            self.points3D.append(point.xyz)
            self.points3D_color.append(point.color)

        self.points3D = torch.tensor(np.array(self.points3D), dtype=torch.float32)
        self.points3D_color = torch.tensor(
            np.array(self.points3D_color), dtype=torch.float32
        )
        self.points3D_color = self.points3D_color / 255.0

        if use_metric3d:
            self.all_normals = []

            for i, image in enumerate(tqdm(self.images)):
                m3d_name = os.path.splitext(image.name)[0] + ".pt"
                m3d = torch.load(os.path.join(metric3d_dir, m3d_name))
                depth = F.interpolate(
                    m3d["depth"][None, None],
                    size=(self.img_wh[1], self.img_wh[0]),
                    mode="nearest",
                )[0, 0]
                normal = F.interpolate(
                    m3d["normal"][None],
                    size=(self.img_wh[1], self.img_wh[0]),
                    mode="bilinear",
                    align_corners=False,
                )[0]
                normal = normal.permute(1, 2, 0)
                confidence = F.interpolate(
                    m3d["confidence"][None, None],
                    size=(self.img_wh[1], self.img_wh[0]),
                    mode="bilinear",
                    align_corners=False,
                )[0, 0]

                self.all_normals.append(normal)

            self.all_normals = torch.stack(self.all_normals)

        else:
            self.all_normals = None
