import fpsample
import plyfile
import numpy as np
from scipy.spatial import KDTree

import torch
import torch.nn as nn
import torch.nn.functional as F
import warp as wp

from .bvh import AABBTree
from .geometry import InterpenetrationFunction, morton_sort
from .rasterize import Rasterizer
from .raytrace import RayTracer
from .color_fn import SphericalVoronoi
from .scheduling import get_cosine_scheduler


def init_points_sfm(data_handler, num_points):
    init_points = data_handler.points3D.float()
    cpu_points = init_points.cpu().numpy()

    sample_points = min(num_points, int(0.95 * init_points.shape[0]))
    print(f"Sampling {sample_points} points from {init_points.shape[0]} sfm points")
    point_inds = fpsample.bucket_fps_kdtree_sampling(cpu_points, sample_points)
    point_inds = torch.tensor(point_inds, dtype=torch.long)

    if num_points < 0.9 * init_points.shape[0]:
        init_points = init_points[point_inds]

    else:
        kdtree = KDTree(cpu_points)
        _, knn_i = kdtree.query(cpu_points, k=6)
        w = torch.rand(num_points - sample_points, 6, device=init_points.device)
        w = w / w.sum(dim=-1, keepdim=True)
        j = torch.randint(0, knn_i.shape[0], (num_points - sample_points,))
        knn_i = torch.tensor(knn_i, dtype=torch.long, device=init_points.device)
        knn_i = knn_i[j]
        _init_points = (w[:, :, None] * init_points[knn_i]).sum(dim=1)

        init_points = torch.cat([init_points[point_inds], _init_points], dim=0)

    return init_points


def init_points_bounded(data_handler, num_points):
    cameras = data_handler.cameras
    alphas = data_handler.alphas
    samples_per_camera = int(num_points / len(cameras)) + 1

    points = torch.empty((0, 3), dtype=torch.float32)
    for i in range(len(cameras)):
        # randomly sample pixels based on alpha
        camera = cameras[i]
        alpha = alphas[i].view(-1)

        pix_inds = torch.multinomial(alpha, samples_per_camera, replacement=False)
        pix_i = (pix_inds // camera.width).float()
        pix_j = (pix_inds % camera.width).float()

        dirs = camera.get_ray_dir(pix_i, pix_j)
        t = torch.rand_like(pix_i) * 10.0 + 3.0  # sample between [3.0, 13.0]
        pts = camera.eye[None, :] + dirs * t[:, None]
        points = torch.cat([points, pts], dim=0)

    return points[:num_points]


def init_points_unbounded(data_handler, num_points):
    cameras = data_handler.cameras
    camera_centers = torch.stack([cam.eye for cam in cameras], dim=0)

    camera_mean = camera_centers.mean(dim=0)
    camera_std = camera_centers.std(dim=0)

    points = torch.randn((num_points, 3), dtype=torch.float32) * camera_std * 3.0
    points = points + camera_mean[None, :]
    return points


class PowerfoamScene(nn.Module):

    def __init__(self, args, attr_dtype="float"):
        super().__init__()
        self.args = args
        if attr_dtype == "float":
            self.attr_dtype = "float"
            self.tscalar = torch.float32
        elif attr_dtype == "half":
            self.attr_dtype = "half"
            self.tscalar = torch.float16

    def initialize_from_dataset(self, train_data_handler, device):

        if self.args.init_type == "sfm":
            if train_data_handler.points3D is None:
                raise ValueError(
                    "No 3D points found in COLMAP output. Cannot use sfm initialization."
                )
            init_points = init_points_sfm(train_data_handler, self.args.init_points)
        elif self.args.init_type == "random_bounded":
            init_points = init_points_bounded(train_data_handler, self.args.init_points)
        elif self.args.init_type == "random_unbounded":
            init_points = init_points_unbounded(
                train_data_handler, self.args.init_points
            )
        else:
            raise ValueError(f"Unknown init_type {self.args.init_type}")

        init_points = init_points.to(self.tscalar).to(device)
        self.points = nn.Parameter(init_points)

        max_radii = 100 * torch.ones(self.points.shape[0])
        for camera in train_data_handler.cameras:
            v = camera.eye[None, :] - self.points.cpu()
            x, y = camera.right, camera.up
            z = torch.cross(x, y, dim=-1)
            v_x = (v * x[None, :]).sum(dim=-1) / x.norm()
            v_y = (v * y[None, :]).sum(dim=-1) / y.norm()
            v_z = (v * z[None, :]).sum(dim=-1) / z.norm()
            max_radii_c = 0.1 * v_z * y.norm()
            mask = v_z > 0
            mask &= (v_x / v_z).abs() < x.norm()
            mask &= (v_y / v_z).abs() < y.norm()
            max_radii[mask] = torch.minimum(max_radii_c[mask], max_radii[mask])

        cpu_points = init_points.cpu().numpy()
        kdtree = KDTree(cpu_points)
        knn_d, _ = kdtree.query(cpu_points, k=8)
        radii = knn_d.mean(axis=1)
        radii = torch.tensor(radii, dtype=self.tscalar, device=device)
        radii = torch.minimum(radii, max_radii.to(device))
        self.radii = nn.Parameter(radii)

        quaternions = torch.randn(
            self.points.shape[0], 4, dtype=self.tscalar, device=device
        )
        quaternions = quaternions / quaternions.norm(dim=-1, keepdim=True)
        self.quaternions = nn.Parameter(quaternions)

        density = (
            torch.ones(self.points.shape[0], dtype=self.tscalar, device=device) * 1e-1
        )
        self.density = nn.Parameter(density)

        texel_sites = (
            torch.randn(
                (
                    self.points.shape[0],
                    self.args.num_texel_sites,
                    2,
                ),
                dtype=self.tscalar,
                device=device,
            )
            * 0.1
        )
        self.texel_sites = nn.Parameter(texel_sites)
        texel_sv_axis = (
            torch.randn(
                (
                    self.points.shape[0],
                    self.args.num_texel_sites,
                    3 * self.args.sv_dof,
                ),
                dtype=self.tscalar,
                device=device,
            )
            * 2.0
        )
        self.texel_sv_axis = nn.Parameter(texel_sv_axis)
        texel_sv_rgb = torch.zeros(
            (
                self.points.shape[0],
                self.args.num_texel_sites,
                3 * self.args.sv_dof,
            ),
            dtype=self.tscalar,
            device=device,
        )
        self.texel_sv_rgb = nn.Parameter(texel_sv_rgb)
        texel_height = torch.zeros(
            (
                self.points.shape[0],
                self.args.num_texel_sites,
            ),
            dtype=self.tscalar,
            device=device,
        )
        self.texel_height = nn.Parameter(texel_height)

        self.aabb_tree = AABBTree(self.device)
        self.rebuild_adjacency()

        self.rasterizer = Rasterizer(self.args, self.device, self.attr_dtype)
        self.raytracer = RayTracer(self.args, self.device, self.attr_dtype)
        self.sv = SphericalVoronoi(
            self.args, self.device, train_data_handler.cameras[0], self.attr_dtype
        )

    @property
    def device(self):
        return self.points.device

    def sort_points(self):
        permutation = morton_sort(self.points)

        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if "env" not in group["name"]:
                stored_state = self.optimizer.state.get(group["params"][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = stored_state["exp_avg"][permutation]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][permutation]

                    del self.optimizer.state[group["params"][0]]
                    group["params"][0] = nn.Parameter(
                        (group["params"][0][permutation].requires_grad_(True))
                    )
                    self.optimizer.state[group["params"][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(
                        group["params"][0][permutation].requires_grad_(True)
                    )
                    optimizable_tensors[group["name"]] = group["params"][0]

        self.points = optimizable_tensors["points"]
        self.radii = optimizable_tensors["radii"]
        self.quaternions = optimizable_tensors["quaternions"]
        self.density = optimizable_tensors["density"]
        self.texel_sites = optimizable_tensors["texel_sites"]
        self.texel_sv_axis = optimizable_tensors["texel_sv_axis"]
        self.texel_sv_rgb = optimizable_tensors["texel_sv_rgb"]
        self.texel_height = optimizable_tensors["texel_height"]

        if hasattr(self, "contrib_ema"):
            self.contrib_ema = self.contrib_ema[permutation]
        if hasattr(self, "point_error_ema"):
            self.point_error_ema = self.point_error_ema[permutation]

    def rebuild_adjacency(self):
        self.aabb_tree.update(self.points, self.get_radii())
        self.adjacency, self.adjacency_offsets = self.aabb_tree.build_cech_complex()

    def get_density(self):
        return F.softplus(self.density, beta=100)

    def get_normals(self):
        quaternions = self.quaternions / self.quaternions.norm(dim=-1, keepdim=True)
        w = quaternions[:, 0]
        x = quaternions[:, 1]
        y = quaternions[:, 2]
        z = quaternions[:, 3]
        normals = torch.stack(
            [
                1 - 2 * (y**2 + z**2),
                2 * (x * y - z * w),
                2 * (x * z + y * w),
            ],
            dim=-1,
        )
        normals = normals / normals.norm(dim=-1, keepdim=True)
        return normals

    def get_tangents(self):
        quaternions = self.quaternions / self.quaternions.norm(dim=-1, keepdim=True)
        w = quaternions[:, 0]
        x = quaternions[:, 1]
        y = quaternions[:, 2]
        z = quaternions[:, 3]
        tangents = torch.stack(
            [
                2 * (x * y + z * w),
                1 - 2 * (x**2 + z**2),
                2 * (y * z - x * w),
            ],
            dim=-1,
        )
        tangents = tangents / tangents.norm(dim=-1, keepdim=True)

        bitangent = torch.stack(
            [
                2 * (x * z - y * w),
                2 * (y * z + x * w),
                1 - 2 * (x**2 + y**2),
            ],
            dim=-1,
        )
        bitangent = bitangent / bitangent.norm(dim=-1, keepdim=True)

        return tangents, bitangent

    def get_radii(self):
        return F.softplus(self.radii, beta=100)

    def get_att_sv(self):
        sv_dof = self.args.sv_dof
        sv_grid_dim = self.args.num_texel_sites

        # Structure of Arrays layout for coalesced access
        # axis: (sv_dof, N * sv_grid_dim, 3)
        # rgb: (sv_dof, N * sv_grid_dim, 3)
        # temp: (sv_dof, N * sv_grid_dim)

        axis = self.texel_sv_axis.view(-1, sv_grid_dim, sv_dof, 3)
        temp = axis.norm(dim=-1)
        axis = axis / axis.norm(dim=-1, keepdim=True)

        axis = axis.view(-1, sv_grid_dim, sv_dof, 3)
        axis = axis.permute(2, 0, 1, 3).reshape(sv_dof, -1, 3).contiguous()

        rgb = self.texel_sv_rgb.view(-1, sv_grid_dim, sv_dof, 3)
        rgb = rgb.permute(2, 0, 1, 3).reshape(sv_dof, -1, 3).contiguous()

        temp = temp.view(-1, sv_grid_dim, sv_dof)
        temp = temp.permute(2, 0, 1).reshape(sv_dof, -1).contiguous()

        return axis, rgb, temp

    def forward(
        self,
        camera,
        depth_quantiles=None,
        ray_gt=None,
        return_point_err=False,
    ):
        normals = self.get_normals()
        tangents, bitangent = self.get_tangents()
        radii = self.get_radii()
        offsets = self.texel_sites * radii[:, None, None]
        offsets = (
            offsets[..., 0:1] * tangents[:, None, :]
            + offsets[..., 1:2] * bitangent[:, None, :]
        )
        texel_sites = self.points[:, None, :] + offsets

        att_sites, att_values, att_temps = self.get_att_sv()
        texel_rgb = self.sv.forward(
            texel_sites.view(-1, 3).detach(), camera, att_sites, att_values, att_temps
        )
        texel_rgb = texel_rgb.view(self.points.shape[0], self.args.num_texel_sites, 3)

        texel_height = self.texel_height * radii[:, None]

        return self.rasterizer.forward(
            camera,
            depth_quantiles,
            self.points,
            self.get_radii(),
            self.get_density(),
            normals,
            texel_sites,
            texel_rgb,
            texel_height,
            self.adjacency,
            self.adjacency_offsets,
            ray_gt,
            return_point_err,
        )

    def interpenetration(self):
        return InterpenetrationFunction.apply(
            self.points,
            self.get_radii(),
            self.adjacency,
            self.adjacency_offsets,
        )

    def declare_optimizers(self, args, iterations):
        params = [
            {
                "params": self.points,
                "lr": args.points_lr_init,
                "name": "points",
            },
            {
                "params": self.density,
                "lr": args.density_lr_init,
                "name": "density",
            },
            {
                "params": self.radii,
                "lr": args.radii_lr_init,
                "name": "radii",
            },
            {
                "params": self.quaternions,
                "lr": args.quaternions_lr_init,
                "name": "quaternions",
            },
            {
                "params": self.texel_sites,
                "lr": args.texel_sites_lr_init,
                "name": "texel_sites",
            },
            {
                "params": self.texel_sv_axis,
                "lr": args.texel_sv_axis_lr_init,
                "name": "texel_sv_axis",
            },
            {
                "params": self.texel_sv_rgb,
                "lr": args.texel_sv_rgb_lr_init,
                "name": "texel_sv_rgb",
            },
            {
                "params": self.texel_height,
                "lr": args.texel_height_lr_init,
                "name": "texel_height",
            },
        ]

        self.optimizer = torch.optim.Adam(params, eps=1e-15)
        self.points_scheduler = get_cosine_scheduler(
            args.points_lr_init,
            args.points_lr_final,
            max_steps=iterations,
        )
        self.density_scheduler = get_cosine_scheduler(
            args.density_lr_init,
            args.density_lr_final,
            warmup_steps=1_000,
            max_steps=iterations,
        )
        self.radii_scheduler = get_cosine_scheduler(
            args.radii_lr_init,
            args.radii_lr_final,
            warmup_steps=1_000,
            max_steps=iterations,
        )
        self.quaternions_scheduler = get_cosine_scheduler(
            args.quaternions_lr_init,
            args.quaternions_lr_final,
            max_steps=iterations,
        )
        self.texel_sites_scheduler = get_cosine_scheduler(
            args.texel_sites_lr_init,
            args.texel_sites_lr_final,
            max_steps=iterations,
        )
        self.texel_sv_axis_scheduler = get_cosine_scheduler(
            args.texel_sv_axis_lr_init,
            args.texel_sv_axis_lr_final,
            max_steps=iterations,
        )
        self.texel_sv_rgb_scheduler = get_cosine_scheduler(
            args.texel_sv_rgb_lr_init,
            args.texel_sv_rgb_lr_final,
            max_steps=iterations,
        )
        self.texel_height_scheduler = get_cosine_scheduler(
            args.texel_height_lr_init,
            args.texel_height_lr_final,
            warmup_steps=2_000,
            max_steps=iterations,
        )

    def update_learning_rate(self, iteration):
        """Learning rate scheduling per step"""
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "points":
                lr = self.points_scheduler(iteration)
                param_group["lr"] = lr
            elif param_group["name"] == "density":
                lr = self.density_scheduler(iteration)
                param_group["lr"] = lr
            elif param_group["name"] == "radii":
                lr = self.radii_scheduler(iteration)
                param_group["lr"] = lr
            elif param_group["name"] == "quaternions":
                lr = self.quaternions_scheduler(iteration)
                param_group["lr"] = lr
            elif param_group["name"] == "texel_sites":
                lr = self.texel_sites_scheduler(iteration)
                param_group["lr"] = lr
            elif param_group["name"] == "texel_sv_axis":
                lr = self.texel_sv_axis_scheduler(iteration)
                param_group["lr"] = lr
            elif param_group["name"] == "texel_sv_rgb":
                lr = self.texel_sv_rgb_scheduler(iteration)
                param_group["lr"] = lr
            elif param_group["name"] == "texel_height":
                lr = self.texel_height_scheduler(iteration)
                param_group["lr"] = lr

    def update_stats(self, contrib, point_error, vis_mask):
        with torch.no_grad():
            vis_mask = vis_mask.float()
            alpha = 0.99 * vis_mask + (1 - vis_mask)
            if hasattr(self, "contrib_ema"):
                self.contrib_ema = alpha * self.contrib_ema + (1 - alpha) * contrib
            else:
                self.contrib_ema = torch.ones_like(contrib) * 1e-5

            alpha = 0.99
            if hasattr(self, "point_error_ema"):
                self.point_error_ema = (
                    alpha * self.point_error_ema + (1 - alpha) * point_error
                )
            else:
                self.point_error_ema = torch.ones_like(point_error) * 1e-5

    def resample(self, target_num_points):
        with torch.no_grad():
            contrib_q = torch.quantile(
                self.contrib_ema, torch.tensor([0.1, 0.99]).to(self.device), dim=0
            )

            contrib_threshold = torch.min(
                torch.tensor(1 / (target_num_points * 25)).to(self.device),
                contrib_q[0],
            )
            contrib_mask = self.contrib_ema > contrib_threshold
            valid_indices = torch.nonzero(contrib_mask, as_tuple=False)[:, 0]

            num_samples = target_num_points - valid_indices.shape[0]
            if num_samples == 0:
                return num_samples

            point_error_q = torch.quantile(self.point_error_ema, 0.99, dim=0)
            prob = self.point_error_ema[valid_indices].clamp(max=point_error_q)

            resample_valid_indices = torch.multinomial(
                prob, num_samples, replacement=False
            )

            duplicate_count = torch.ones_like(valid_indices)
            duplicate_count.index_add_(
                0, resample_valid_indices, torch.ones_like(resample_valid_indices)
            )
            duplicate_mask = duplicate_count > 1

            resample_cell_indices = valid_indices[resample_valid_indices]
            resample_duplicate_mask = duplicate_mask[resample_valid_indices]
            resample_duplicate_count = duplicate_count[resample_valid_indices]

            new_indices = torch.cat([valid_indices, resample_cell_indices], dim=0)
            new_duplicate_mask = torch.cat(
                [duplicate_mask, resample_duplicate_mask], dim=0
            ).bool()
            new_duplicate_count = torch.cat(
                [duplicate_count, resample_duplicate_count], dim=0
            )

            optimizable_tensors = {}
            for group in self.optimizer.param_groups:
                stored_state = self.optimizer.state.get(group["params"][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = stored_state["exp_avg"][new_indices]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][new_indices]

                    del self.optimizer.state[group["params"][0]]
                    group["params"][0] = nn.Parameter(
                        (group["params"][0][new_indices].requires_grad_(True))
                    )
                    self.optimizer.state[group["params"][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(
                        group["params"][0][new_indices].requires_grad_(True)
                    )
                    optimizable_tensors[group["name"]] = group["params"][0]

            self.points = optimizable_tensors["points"]
            self.density = optimizable_tensors["density"]
            self.radii = optimizable_tensors["radii"]
            self.quaternions = optimizable_tensors["quaternions"]
            self.texel_sites = optimizable_tensors["texel_sites"]
            self.texel_sv_axis = optimizable_tensors["texel_sv_axis"]
            self.texel_sv_rgb = optimizable_tensors["texel_sv_rgb"]
            self.texel_height = optimizable_tensors["texel_height"]

            # Sample a direction perpendicular to the normals and perturb
            normals = self.get_normals()
            cell_radius = self.get_radii()
            direction = torch.randn_like(normals)
            direction = direction - (direction * normals).sum(dim=-1, keepdim=True)
            direction = direction / direction.norm(dim=-1, keepdim=True)
            perturbation = 0.05 * cell_radius[..., None] * direction

            self.points[new_duplicate_mask] += perturbation[new_duplicate_mask]

            # Propagate running contribution averages
            self.contrib_ema = self.contrib_ema[new_indices]
            self.contrib_ema /= new_duplicate_count.float()
            self.point_error_ema = self.point_error_ema[new_indices]
            self.point_error_ema /= new_duplicate_count.float()

        return num_samples

    def save_pt(self, pt_path):
        points = self.points.detach().float().cpu()
        density = self.density.detach().float().cpu()
        radii = self.radii.detach().float().cpu()
        quaternions = self.quaternions.detach().float().cpu()
        texel_sites = self.texel_sites.detach().float().cpu()
        texel_sv_axis = self.texel_sv_axis.detach().float().cpu()
        texel_sv_rgb = self.texel_sv_rgb.detach().float().cpu()
        texel_height = self.texel_height.detach().float().cpu()
        adjacency = self.adjacency.cpu()
        adjacency_offsets = self.adjacency_offsets.cpu()

        scene_data = {
            "points": points,
            "density": density,
            "radii": radii,
            "quaternions": quaternions,
            "texel_sites": texel_sites,
            "texel_sv_axis": texel_sv_axis,
            "texel_sv_rgb": texel_sv_rgb,
            "texel_height": texel_height,
            "adjacency": adjacency,
            "adjacency_offsets": adjacency_offsets,
        }
        torch.save(scene_data, pt_path)

    def load_pt(self, pt_path):
        scene_data = torch.load(pt_path)

        self.points.data = scene_data["points"].to(self.device)
        self.density.data = scene_data["density"].to(self.tscalar).to(self.device)
        self.radii.data = scene_data["radii"].to(self.tscalar).to(self.device)
        self.quaternions.data = (
            scene_data["quaternions"].to(self.tscalar).to(self.device)
        )
        self.texel_sites.data = (
            scene_data["texel_sites"].to(self.tscalar).to(self.device)
        )
        self.texel_sv_axis.data = (
            scene_data["texel_sv_axis"].to(self.tscalar).to(self.device)
        )
        self.texel_sv_rgb.data = (
            scene_data["texel_sv_rgb"].to(self.tscalar).to(self.device)
        )
        self.texel_height.data = (
            scene_data["texel_height"].to(self.tscalar).to(self.device)
        )

        self.adjacency = scene_data["adjacency"].to(self.device)
        self.adjacency_offsets = scene_data["adjacency_offsets"].to(self.device)

    def save_pc(self, ply_path):
        points = self.points.detach().float().cpu()
        normals = self.get_normals().detach().float().cpu()
        colors = 0.5 + self.texel_sv_rgb.detach().float().cpu().view(
            -1, self.args.sv_dof, 3
        ).mean(dim=1)
        colors = colors.clip(0.0, 1.0)

        verts = np.array(
            list(
                (
                    p[0].item(),
                    p[1].item(),
                    p[2].item(),
                    n[0].item(),
                    n[1].item(),
                    n[2].item(),
                    255 * c[0].item(),
                    255 * c[1].item(),
                    255 * c[2].item(),
                )
                for p, n, c in zip(points, normals, colors)
            ),
            dtype=[
                ("x", "f4"),
                ("y", "f4"),
                ("z", "f4"),
                ("nx", "f4"),
                ("ny", "f4"),
                ("nz", "f4"),
                ("red", "u1"),
                ("green", "u1"),
                ("blue", "u1"),
            ],
        )
        ply = plyfile.PlyData([plyfile.PlyElement.describe(verts, "vertex")], text=True)
        ply.write(ply_path)
