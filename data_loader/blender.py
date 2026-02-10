import os
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import json
import math

from powerfoam.camera import TorchCamera
from .metric3D import Metric3DEstimator


class BlenderDataset(Dataset):

    def __init__(self, datadir, split, downsample, alpha_format_on_disk, use_metric3d):

        self.root_dir = datadir
        self.split = split
        self.downsample = downsample
        self.alpha_format_on_disk = alpha_format_on_disk
        self.use_metric3d = use_metric3d

        self.blender2opencv = np.array(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        )

        with open(
            os.path.join(self.root_dir, f"transforms_{self.split}.json"), "r"
        ) as f:
            meta = json.load(f)
        if "w" in meta and "h" in meta:
            W, H = int(meta["w"]), int(meta["h"])
        else:
            W, H = 800, 800

        metric3d_dir = os.path.join(datadir, f"metric3d/{self.split}")
        if use_metric3d and not os.path.exists(metric3d_dir):
            print("Precomputed Metric3D data not found; running now...")
            os.makedirs(metric3d_dir)
            input_paths = list(
                os.path.join(self.root_dir, f"{frame['file_path']}.png")
                for _, frame in enumerate(meta["frames"])
            )
            estimator = Metric3DEstimator()
            estimator.process_dir(input_paths, metric3d_dir)

        self.img_wh = (int(W / self.downsample), int(H / self.downsample))
        w, h = self.img_wh

        focal = 0.5 * w / math.tan(0.5 * meta["camera_angle_x"])  # scaled focal length

        self.intrinsics = torch.tensor(
            [[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]]
        )

        # Compute camera space basis vectors (similar to image_plane_basis in colmap.py)
        # For pixel (0.5, 0.5), compute the direction in camera space
        pix_min = np.array([0.5, 0.5], dtype=np.float32)
        # Convert pixel to camera space direction
        cam_x = (pix_min[0] - w / 2) / focal
        cam_y = (pix_min[1] - h / 2) / focal
        # Camera space basis vectors
        cam_space_right = torch.tensor([-cam_x, 0.0, 0.0], dtype=torch.float32)
        cam_space_up = torch.tensor([0.0, cam_y, 0.0], dtype=torch.float32)

        self.poses = []
        self.all_cameras = []
        self.all_rgbs = []
        self.all_alphas = []
        for i, frame in enumerate(meta["frames"]):
            pose = np.array(frame["transform_matrix"]) @ self.blender2opencv
            c2w = torch.FloatTensor(pose)
            self.poses.append(c2w)
            world_right = torch.einsum("j,kj->k", cam_space_right, c2w[:3, :3])
            world_up = torch.einsum("j,kj->k", cam_space_up, c2w[:3, :3])

            self.all_cameras.append(
                TorchCamera(
                    eye=c2w[:3, 3].pin_memory(),
                    right=world_right.pin_memory(),
                    up=world_up.pin_memory(),
                    width=self.img_wh[0],
                    height=self.img_wh[1],
                )
            )

            im = Image.open(os.path.join(self.root_dir, f"{frame['file_path']}.png"))
            if self.downsample != 1.0:
                im = im.resize(self.img_wh, Image.LANCZOS)
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

        self.points3D = None
        self.points3D_color = None

        if self.use_metric3d:
            self.all_normals = []

            for i, frame in enumerate(meta["frames"]):
                filename = os.path.basename(frame["file_path"])
                m3d_name = os.path.join(metric3d_dir, f"{filename}.pt")
                m3d = torch.load(m3d_name)
                depth = F.interpolate(
                    m3d["depth"][None, None],
                    size=(h, w),
                    mode="nearest",
                )[0, 0]
                normal = F.interpolate(
                    m3d["normal"][None],
                    size=(h, w),
                    mode="bilinear",
                )[0]
                normal = normal.permute(1, 2, 0)
                confidence = F.interpolate(
                    m3d["confidence"][None, None],
                    size=(h, w),
                    mode="bilinear",
                )[0, 0]

                normal = torch.einsum(
                    "ij,kj->ik", normal.reshape(-1, 3), self.poses[i][:3, :3]
                )
                normal = normal.reshape(h, w, 3)

                self.all_normals.append(normal)

            self.all_normals = torch.stack(self.all_normals)

        else:
            self.all_normals = None
