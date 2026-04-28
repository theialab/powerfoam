import os
import numpy as np
import torch
from tqdm import tqdm
import configargparse
import open3d as o3d

from configs import *
from data_loader import DataHandler
from powerfoam.scene import PowerfoamScene
from powerfoam.camera import TorchCamera
from render import to_cam_open3d  # reuse intrinsics/extrinsics conversion
import warp as wp


def _make_rgbd(rgb, depth, alpha, depth_trunc=20.0, min_depth=1e-4):
    if rgb.dim() == 3 and rgb.shape[-1] == 3:
        rgb_np = (rgb.clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
    elif rgb.dim() == 3 and rgb.shape[0] == 3:
        rgb_np = (rgb.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
    else:
        raise ValueError(f"Unexpected rgb shape: {tuple(rgb.shape)}")

    depth_np = depth.squeeze(-1).cpu().numpy().astype(np.float32) if depth.ndim == 3 \
        else depth.cpu().numpy().astype(np.float32)
    # depth_np[~np.isfinite(depth_np)] = 0.0
    # depth_np[depth_np <= float(min_depth)] = 0.0
    # if depth_trunc is not None:
    #     depth_np[depth_np > float(depth_trunc)] = 0.0

    # if alpha is not None:
    #     alpha_np = alpha.cpu().numpy()
    #     depth_np[alpha_np < 0.5] = 0.0

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(rgb_np),
        o3d.geometry.Image(depth_np),
        depth_trunc=depth_trunc,
        convert_rgb_to_intensity=False,
        depth_scale=1.0,
    )
    return rgbd


def compute_pairwise_alignment(pcd_a: o3d.geometry.PointCloud, pcd_b: o3d.geometry.PointCloud):
    # distances from a to nearest in b, and from b to nearest in a
    d_ab = np.asarray(pcd_a.compute_point_cloud_distance(pcd_b))
    d_ba = np.asarray(pcd_b.compute_point_cloud_distance(pcd_a))
    stats = {
        'a->b_mean': float(d_ab.mean()) if d_ab.size else float('nan'),
        'a->b_p95': float(np.percentile(d_ab, 95)) if d_ab.size else float('nan'),
        'b->a_mean': float(d_ba.mean()) if d_ba.size else float('nan'),
        'b->a_p95': float(np.percentile(d_ba, 95)) if d_ba.size else float('nan'),
    }
    return stats


def main():
    parser = configargparse.ArgParser()
    get_params = add_group(parser, Params)
    parser.add_argument('-c', '--config', is_config_file=True, help='Path to config file')
    parser.add_argument('--frames', type=int, default=5, help='Number of test frames')
    parser.add_argument('--voxel', type=float, default=0.01, help='Downsample voxel size for PCDs')
    parser.add_argument('--depth_trunc', type=float, default=20.0)
    parser.add_argument('--min_depth', type=float, default=1e-4)
    args = parser.parse_args()

    params = get_params(args)

    wp.init()
    test_data_handler = DataHandler(params)
    test_data_handler.reload('test', downsample=params.downsample[-1])

    model = PowerfoamScene(params)
    model.initialize_from_dataset(test_data_handler, device='cuda')
    checkpoint = args.config.replace('/config.yaml', '')
    model.load_pt(f"{checkpoint}/model.pt")

    cams = test_data_handler.cameras
    cam_params = to_cam_open3d(cams)

    pcs = []
    step = len(cams) // args.frames if args.frames < len(cams) else 1
    with torch.no_grad():
        for i in tqdm(range(0, len(cams), step), desc='Build PCDs'):
            camera = cams[i]
            rgb_gt = test_data_handler.rgbs[i].cuda()
            depth_quantile = 0.5 * torch.ones(*rgb_gt.shape[:-1], 1, device=model.device)
            result = model.forward(camera, depth_quantiles=depth_quantile)
            rgb = result[0]
            depth = result[4]
            alpha = result[1]

            rgbd = _make_rgbd(rgb, depth, alpha, depth_trunc=args.depth_trunc, min_depth=args.min_depth)
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd,
                intrinsic=cam_params[i].intrinsic,
                extrinsic=cam_params[i].extrinsic,
            )
            if args.voxel > 0:
                pcd = pcd.voxel_down_sample(args.voxel)
            o3d.io.write_point_cloud(f"pcd_{i}.ply", pcd)
            pcs.append(pcd)

    # Pairwise adjacent alignment stats
    for i in range(len(pcs) - 1):
        stats = compute_pairwise_alignment(pcs[i], pcs[i+1])
        print(f"pair {i}->{i+1}: mean {stats['a->b_mean']:.4f} (95% {stats['a->b_p95']:.4f}), "
              f"rev mean {stats['b->a_mean']:.4f} (95% {stats['b->a_p95']:.4f})")

    # Save merged
    if pcs:
        merged = pcs[0]
        for i in range(1, len(pcs)):
            merged += pcs[i]
        o3d.io.write_point_cloud("pcd_merged.ply", merged)
        print("Saved merged point cloud to pcd_merged.ply")


if __name__ == '__main__':
    main()
