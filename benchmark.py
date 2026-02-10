import os
import numpy as np
from scipy.spatial import ConvexHull, cKDTree
from PIL import Image
import configargparse
import warnings

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import warp as wp

from configs import *
from data_loader import DataHandler
from powerfoam.scene import PowerfoamScene
from powerfoam.color_fn import SphericalVoronoi
from powerfoam.metrics import psnr, ssim


seed = 0
torch.random.manual_seed(seed)
np.random.seed(seed)


def build_power_adjacency(
    points: torch.Tensor, radii: torch.Tensor, alpha_complex=False
):
    # 1. Lift points to 4D
    # The 4th coord is: x^2 + y^2 + z^2 - r^2
    norms = points.norm(dim=-1)
    weights = norms**2 - radii**2
    lifted_points = torch.cat([points, weights[:, None]], dim=-1)

    # 2. Compute Convex Hull in 4D
    lifted_points_np = lifted_points.detach().cpu().numpy()
    hull = ConvexHull(lifted_points_np)

    # 3. Filter for "Lower Hull"
    # hull.equations is (N_faces, 5) -> [nx, ny, nz, nw, offset]
    # We keep simplices where the normal points 'down' in the 4th dim (nw < 0)
    equations = hull.equations
    simplices = hull.simplices

    # Filter simplices => weighted delaunay tetrahedrons
    is_lower = equations[:, 3] < 0
    lower_hull_simplices = torch.tensor(simplices[is_lower], device=points.device)

    # Get unique edges from tetrahedrons
    edges = lower_hull_simplices[:, [0, 1]]
    edges = torch.cat([edges, lower_hull_simplices[:, [1, 2]]], dim=0)
    edges = torch.cat([edges, lower_hull_simplices[:, [2, 0]]], dim=0)
    edges = torch.cat([edges, lower_hull_simplices[:, [0, 3]]], dim=0)
    edges = torch.cat([edges, lower_hull_simplices[:, [1, 3]]], dim=0)
    edges = torch.cat([edges, lower_hull_simplices[:, [2, 3]]], dim=0)
    unique_edges = torch.unique(edges, dim=0)

    if alpha_complex:
        edge_norm = torch.norm(
            points[unique_edges[:, 0]] - points[unique_edges[:, 1]], dim=-1
        )
        radii_sum = radii[unique_edges[:, 0]] + radii[unique_edges[:, 1]]
        mask = edge_norm < radii_sum
        unique_edges = unique_edges[mask]

    # Count edges per point and build adjacency list vectorially
    directed_edges = torch.cat([unique_edges, unique_edges.flip(1)], dim=0)

    # Sort by source vertex
    sort_idx = torch.argsort(directed_edges[:, 0])
    sorted_edges = directed_edges[sort_idx]

    # Flattened adjacency list
    adjacency = sorted_edges[:, 1]

    # Compute offsets
    sources = sorted_edges[:, 0]
    counts = torch.bincount(sources, minlength=points.shape[0])
    offsets = torch.cat(
        (torch.tensor([0], device=points.device), torch.cumsum(counts, dim=0))
    )
    adjacency_offsets = offsets.int()

    return adjacency.to(points.device), adjacency_offsets.to(points.device)


def inverse_softplus(x, beta=100, threshold=20):
    mask = x * beta > threshold
    y = torch.empty_like(x)
    y[mask] = x[mask]
    y[~mask] = torch.log(torch.exp(beta * x[~mask]) - 1) / beta
    return y


def get_steiner_points(points, radii, cameras):
    points_mean = points.mean(dim=0, keepdim=True)
    points_std = points.std(dim=0, keepdim=True)

    steiner_points = torch.empty(0, 3, device=points.device)
    steiner_radii = torch.empty(0, device=points.device)
    for i in range(10):
        all_points = torch.cat([points, steiner_points], dim=0)
        all_radii = torch.cat([radii, steiner_radii], dim=0)
        kdtree = cKDTree(all_points.cpu().numpy())

        num_random_points = int(0.25 * points.shape[0])
        sample_points = points_mean + 0.5 * points_std * torch.randn(
            num_random_points, 3, device=points.device
        )
        dists, idxs = kdtree.query(sample_points.cpu().numpy(), k=32)
        dists = torch.from_numpy(dists).to(points.device).float()
        idxs = torch.from_numpy(idxs).to(points.device)
        sample_radii, closest_idx = (dists - all_radii[idxs]).min(dim=1)
        closest_pt_idx = idxs[torch.arange(idxs.shape[0]), closest_idx]
        ratio = sample_radii / all_radii[closest_pt_idx]

        mask = (ratio > 2.0) & (ratio < 6.0)
        sample_points = sample_points[mask]
        sample_radii = sample_radii[mask]

        # Select candidates from samples so that they have minimum overlap with each other
        is_candidate = torch.zeros(
            sample_points.shape[0], dtype=torch.bool, device=points.device
        )
        idx = torch.randint(0, sample_points.shape[0], (1,), device=points.device)
        is_candidate[idx] = True
        while True:
            if is_candidate.all():
                break

            candidates = sample_points[is_candidate]
            candidates_radii = sample_radii[is_candidate]
            remaining_indices = torch.nonzero(~is_candidate, as_tuple=True)[0]
            remaining_samples = sample_points[~is_candidate]
            remaining_radii = sample_radii[~is_candidate]
            dists = torch.norm(candidates[:, None] - remaining_samples[None, :], dim=-1)
            overlap = (candidates_radii[:, None] + remaining_radii[None, :] - dists) / (
                candidates_radii[:, None] + remaining_radii[None, :]
            )
            max_overlap, _ = overlap.max(dim=0)
            idx = max_overlap.argmin()
            if max_overlap[idx] > 0.1:
                break

            is_candidate[remaining_indices[idx]] = True

        steiner_points = torch.cat([steiner_points, sample_points[is_candidate]], dim=0)
        steiner_radii = torch.cat(
            [steiner_radii, 0.8 * sample_radii[is_candidate]], dim=0
        )

    steiner_points = steiner_points.to(points.dtype)
    steiner_radii = inverse_softplus(steiner_radii, beta=100)
    steiner_radii = steiner_radii.to(radii.dtype)

    return steiner_points, steiner_radii


def add_steiner_points(model, new_points, new_radii, attr_dtype):
    num_new_samples = new_points.shape[0]
    new_params = {
        "points": new_points,
        "radii": new_radii,
        "density": -10
        * torch.ones(num_new_samples, device=model.device, dtype=attr_dtype),
        "quaternions": torch.randn(
            num_new_samples, 4, device=model.device, dtype=attr_dtype
        ),
        "texel_sites": torch.zeros(
            num_new_samples,
            *model.texel_sites.shape[1:],
            device=model.device,
            dtype=attr_dtype,
        ),
        "texel_sv_axis": torch.randn(
            num_new_samples,
            *model.texel_sv_axis.shape[1:],
            device=model.device,
            dtype=attr_dtype,
        ),
        "texel_sv_rgb": torch.zeros(
            num_new_samples,
            *model.texel_sv_rgb.shape[1:],
            device=model.device,
            dtype=attr_dtype,
        ),
        "texel_height": torch.zeros(
            num_new_samples,
            *model.texel_height.shape[1:],
            device=model.device,
            dtype=attr_dtype,
        ),
    }
    optimizable_tensors = {}
    for group in model.optimizer.param_groups:
        stored_state = model.optimizer.state.get(group["params"][0], None)
        if stored_state is not None:
            new_stored_state = torch.zeros_like(stored_state[:num_new_samples])
            stored_state["exp_avg"] = torch.cat(
                [stored_state["exp_avg"], new_stored_state["exp_avg"]], dim=0
            )
            stored_state["exp_avg_sq"] = torch.cat(
                [stored_state["exp_avg_sq"], new_stored_state["exp_avg_sq"]], dim=0
            )

            del model.optimizer.state[group["params"][0]]
            group["params"][0] = nn.Parameter(
                torch.cat(
                    [group["params"][0], new_params[group["name"]]], dim=0
                ).requires_grad_(True)
            )
            model.optimizer.state[group["params"][0]] = stored_state
            optimizable_tensors[group["name"]] = group["params"][0]
        else:
            group["params"][0] = nn.Parameter(
                torch.cat(
                    [group["params"][0], new_params[group["name"]]], dim=0
                ).requires_grad_(True)
            )
            optimizable_tensors[group["name"]] = group["params"][0]

    model.points = optimizable_tensors["points"]
    model.density = optimizable_tensors["density"]
    model.radii = optimizable_tensors["radii"]
    model.quaternions = optimizable_tensors["quaternions"]
    model.texel_sites = optimizable_tensors["texel_sites"]
    model.texel_sv_axis = optimizable_tensors["texel_sv_axis"]
    model.texel_sv_rgb = optimizable_tensors["texel_sv_rgb"]
    model.texel_height = optimizable_tensors["texel_height"]


def test(args, config_path, render_type):
    wp.init()

    checkpoint = config_path.replace("/config.yaml", "")
    os.makedirs(os.path.join(checkpoint, "test"), exist_ok=True)

    train_data_handler = DataHandler(args)
    train_data_handler.reload("train", downsample=args.downsample[-1])
    test_data_handler = DataHandler(args)
    test_data_handler.reload("test", downsample=args.downsample[-1])

    # Setting up model
    model = PowerfoamScene(args, attr_dtype="half")
    model.initialize_from_dataset(test_data_handler, device="cuda")
    model.load_pt(f"{checkpoint}/model.pt")
    model.declare_optimizers(args, args.iterations)
    model.sort_points()

    if render_type == "raytrace":
        with torch.no_grad():
            steiner_points, steiner_radii = get_steiner_points(
                model.points,
                model.get_radii().to(torch.float32),
                test_data_handler.cameras,
            )
            add_steiner_points(
                model,
                steiner_points,
                steiner_radii,
                model.tscalar,
            )
            model.sort_points()

    with torch.no_grad():
        points = model.points
        radii = model.get_radii()

        adjacency, adjacency_offsets = build_power_adjacency(
            points, radii, alpha_complex=(render_type == "rasterize")
        )
        num_adjs = adjacency_offsets.diff()

        pm = 0.5 * (points.norm(dim=-1) ** 2 - radii**2)
        self_points = points.repeat_interleave(num_adjs, dim=0)
        adjacency_diff = points[adjacency] - self_points
        pm_diff = pm[adjacency] - pm.repeat_interleave(num_adjs, dim=0)
        adjacency_diff = torch.cat([adjacency_diff, pm_diff[:, None]], dim=-1)
        adjacency_diff = adjacency_diff.to(torch.float16)

        density = model.get_density()
        normals = model.get_normals()
        tangents, bitangent = model.get_tangents()
        offsets = model.texel_sites * radii[:, None, None]
        offsets = (
            offsets[..., 0:1] * tangents[:, None, :]
            + offsets[..., 1:2] * bitangent[:, None, :]
        )
        texel_sites = model.points[:, None, :] + offsets

        att_sites, att_values, att_temps = model.get_att_sv()

        texel_height = model.texel_height * radii[:, None]

        cameras = test_data_handler.cameras
        n_frames = len(cameras)
        start_point_idxs = []
        for i in range(n_frames):
            camera_eye = cameras[i].eye.to(model.device)
            dists = torch.linalg.norm(points - camera_eye[None, :], dim=-1)
            start_point_idxs.append(int(torch.argmin(dists**2 - radii**2)))

        # Warmup
        print("Warming up...")
        torch.cuda.synchronize()

        for i in range(n_frames):
            camera = cameras[i]
            start_point_idx = start_point_idxs[i]
            texel_rgb = model.sv.forward(
                texel_sites.view(-1, 3).detach(),
                camera,
                att_sites,
                att_values,
                att_temps,
            )
            texel_rgb = texel_rgb.view(model.points.shape[0], args.num_texel_sites, 3)

            if render_type == "rasterize":
                rgb, _, _ = model.rasterizer.benchmark(
                    camera,
                    points,
                    radii,
                    density,
                    normals,
                    texel_sites,
                    texel_rgb,
                    texel_height,
                    adjacency,
                    adjacency_offsets,
                    adjacency_diff,
                    1e-2,
                )
            elif render_type == "raytrace":
                rgb, _, _ = model.raytracer.benchmark(
                    camera,
                    start_point_idx,
                    points,
                    radii,
                    density,
                    normals,
                    texel_sites,
                    texel_rgb,
                    texel_height,
                    adjacency,
                    adjacency_offsets,
                    adjacency_diff,
                    1e-2,
                )

        torch.cuda.synchronize()

        # Actual benchmarking
        print("Benchmarking...")
        n_reps = 5
        output = torch.empty(
            (n_frames, cameras[0].height, cameras[0].width, 3), device=model.device
        )
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        for _ in range(n_reps):
            for i in range(n_frames):
                camera = cameras[i]
                start_point_idx = start_point_idxs[i]
                texel_rgb = model.sv.forward(
                    texel_sites.view(-1, 3).detach(),
                    camera,
                    att_sites,
                    att_values,
                    att_temps,
                )
                texel_rgb = texel_rgb.view(
                    model.points.shape[0], args.num_texel_sites, 3
                )

                if render_type == "rasterize":
                    rgb, _, _ = model.rasterizer.benchmark(
                        camera,
                        points,
                        radii,
                        density,
                        normals,
                        texel_sites,
                        texel_rgb,
                        texel_height,
                        adjacency,
                        adjacency_offsets,
                        adjacency_diff,
                        1e-2,
                    )
                    output[i] = rgb
                elif render_type == "raytrace":
                    rgb, _, _ = model.raytracer.benchmark(
                        camera,
                        start_point_idx,
                        points,
                        radii,
                        density,
                        normals,
                        texel_sites,
                        texel_rgb,
                        texel_height,
                        adjacency,
                        adjacency_offsets,
                        adjacency_diff,
                        1e-2,
                    )
                    output[i] = rgb

        end_event = torch.cuda.Event(enable_timing=True)
        end_event.record()
        torch.cuda.synchronize()

        total_time = start_event.elapsed_time(end_event)
        framerate = n_reps * n_frames / (total_time / 1000.0)

        print(f"Total time: {total_time} ms")
        print(f"FPS: {framerate}")

        psnr_list = []
        for i in range(n_frames):
            rgb = output[i]
            rgb_gt = test_data_handler.rgbs[i].cuda()
            psnr_list.append(psnr(rgb, rgb_gt))

        print(f"PSNR: {sum(psnr_list) / len(psnr_list)}")


if __name__ == "__main__":
    parser = configargparse.ArgParser()

    get_params = add_group(parser, Params)

    # Add argument to specify a custom config file
    parser.add_argument(
        "-c", "--config", is_config_file=True, help="Path to config file"
    )

    # Add argument to specify render type
    parser.add_argument(
        "--render_type",
        type=str,
        default="rasterize",
        choices=["raytrace", "rasterize"],
        help="Render type: raytrace or rasterize",
    )

    # Parse arguments
    args = parser.parse_args()

    test(get_params(args), args.config, args.render_type)
