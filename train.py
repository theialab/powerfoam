import os
import uuid
import yaml
import configargparse
import numpy as np
import tqdm
from PIL import Image
from matplotlib import cm
import warp as wp
import gc

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from configs import *
from data_loader import DataHandler
from powerfoam.scene import PowerfoamScene
from powerfoam.geometry import normals_from_depth, depth_bilateral_filter
from powerfoam.scheduling import get_exp_scheduler, get_cosine_scheduler
from powerfoam.metrics import psnr, ssim

torch.manual_seed(42)
np.random.seed(42)


def train(args):
    wp.init()

    # Setting up output directory
    if not args.dry_run:
        if len(args.experiment_name) == 0:
            unique_str = str(uuid.uuid4())[:8]
            experiment_name = f"{args.scene}@{unique_str}"
        else:
            experiment_name = args.experiment_name
        print("Experiment Name:", experiment_name)
        out_dir = f"output/{experiment_name}"
        writer = SummaryWriter(out_dir, purge_step=0)
        os.makedirs(f"{out_dir}/test", exist_ok=True)

        def represent_list_inline(dumper, data):
            return dumper.represent_sequence(
                "tag:yaml.org,2002:seq", data, flow_style=True
            )

        yaml.add_representer(list, represent_list_inline)

        # Save the arguments to a YAML file
        with open(f"{out_dir}/config.yaml", "w") as yaml_file:
            yaml.dump(vars(args), yaml_file, default_flow_style=False)

    # Setting up dataloader
    test_data_handler = DataHandler(args)
    test_data_handler.reload("test", downsample=args.downsample[-1])
    train_data_handler = DataHandler(args)
    train_data_handler.reload("train", downsample=args.downsample[0])
    train_data_iter = train_data_handler.get_iter()
    print("Loaded dataset")

    # Setting up model
    model = PowerfoamScene(args)
    model.initialize_from_dataset(train_data_handler, device="cuda")
    model.declare_optimizers(args, args.iterations)
    model.sort_points()
    args.init_points = model.points.shape[0]

    def test_loop(step, final=False):
        psnr_list = []
        with torch.no_grad():
            num_test = len(test_data_handler.cameras)
            for i in range(num_test):
                camera = test_data_handler.cameras[i]
                rgb_gt = test_data_handler.rgbs[i].cuda()

                depth_quantile = 0.5 * torch.ones(
                    *rgb_gt.shape[:-1], 1, device=model.device
                )

                result = model.forward(camera, depth_quantiles=depth_quantile)
                rgb = result[0]
                alpha = result[1]
                normal = result[3]
                depth = result[4]

                img_psnr = psnr(rgb, rgb_gt)
                psnr_list.append(img_psnr.item())

                if not args.dry_run:
                    if i % (num_test // 3) != 0 and not final:
                        continue

                    error = (rgb - rgb_gt).abs()
                    error = (error.cpu() * 255).to(torch.uint8)
                    rgb_output = (rgb.cpu().clamp(min=0, max=1) * 255).to(torch.uint8)
                    rgb_gt = (rgb_gt.cpu() * 255).to(torch.uint8)
                    im_top = torch.hstack((rgb_gt, rgb_output, error))

                    alpha = (
                        (alpha.cpu() * 255).to(torch.uint8)[:, :, None].repeat(1, 1, 3)
                    )
                    normal = ((normal + 1) * 0.5 * 255).cpu().to(torch.uint8)
                    depth = depth.cpu()[:, :, 0]
                    depth_min = depth[depth > 0].min()
                    depth_max = depth.max()
                    depth = (depth - depth_min) / (depth_max - depth_min + 1e-8)
                    depth = (cm.viridis(depth.numpy())[:, :, :3] * 255).astype(np.uint8)
                    depth = torch.from_numpy(depth)
                    im_bottom = torch.hstack((alpha, normal, depth))

                    im = torch.vstack((im_top, im_bottom))

                    # if not final and step % 400 == 399:
                    #     writer.add_image(f"test/{i:03d}", im.permute(2, 0, 1), step)

                    if final:
                        im = Image.fromarray(im.numpy())
                        im.save(f"{out_dir}/test/{i:03d}.png")

        average_psnr = sum(psnr_list) / len(psnr_list)
        if final and not args.dry_run:
            f = open(f"{out_dir}/metrics.txt", "w")
            f.write(f"Average PSNR: {average_psnr}")
            f.close()

        if not args.dry_run:
            writer.add_scalar("test/psnr", average_psnr, step)

    def train_loop():
        nonlocal train_data_iter

        print("Starting training")

        train_psnr_ema = 0.0
        ema_weight = 0.0

        iters_since_triangulation = 0
        triangulation_interval = 1

        normal_loss_scheduler = get_exp_scheduler(
            args.normal_weight, 1e-1 * args.normal_weight, args.iterations
        )
        contrib_loss_scheduler = get_exp_scheduler(
            args.contribution_weight, 1e-3 * args.contribution_weight, args.iterations
        )
        interpenetration_loss_scheduler = get_exp_scheduler(
            args.interpenetration_weight,
            1e-3 * args.interpenetration_weight,
            args.iterations,
        )
        sigma_spatial_scheduler = get_cosine_scheduler(4.5, 1.5, args.iterations)

        with tqdm.trange(args.iterations, desc="Training") as train:
            for i in train:
                if i and i in args.downsample_iterations:
                    downsample_idx = args.downsample_iterations.index(i)
                    downsample = args.downsample[downsample_idx]
                    train_data_handler.reload("train", downsample=downsample)
                    train_data_iter = train_data_handler.get_iter()

                torch.cuda.nvtx.range_push("Train Step")
                torch.cuda.nvtx.range_push("Loading Data")

                camera, rgb_gt, alpha_gt, normal_gt = next(train_data_iter)
                random_bkgd = torch.rand_like(rgb_gt)
                rgb_gt += (1 - alpha_gt[..., None]) * random_bkgd

                torch.cuda.nvtx.range_pop()  # Loading Data
                torch.cuda.nvtx.range_push("Zero Grad")

                model.optimizer.zero_grad(set_to_none=True)

                torch.cuda.nvtx.range_pop()  # Zero Grad
                torch.cuda.nvtx.range_push("Rebuild Adjacency")

                iters_since_triangulation += 1
                if iters_since_triangulation % triangulation_interval == 0:
                    model.rebuild_adjacency()
                    iters_since_triangulation = 0
                    triangulation_interval += 1
                    triangulation_interval = min(triangulation_interval, 20)

                torch.cuda.nvtx.range_pop()  # Rebuild Adjacency
                torch.cuda.nvtx.range_push("Forward")

                # depth_quantiles = (
                #     torch.rand(*rgb_gt.shape[:-1], 1, device=model.device)
                # ).clip(min=1e-8)
                # depth_quantiles = (
                #     torch.cat(
                #         [depth_quantiles, 0.5 * torch.ones_like(depth_quantiles)],
                #         dim=-1,
                #     )
                #     .sort(dim=-1)
                #     .values
                # )
                # mask = (depth_quantiles - 0.5).abs().argmin(dim=-1)

                result = model.forward(
                    camera,
                    # depth_quantiles=depth_quantiles,
                    ray_gt=rgb_gt,
                    return_point_err=True,
                )
                rgb = result[0]
                alpha = result[1]
                normal_err = result[2]
                normal = result[3]
                depth = result[4]
                # err = result[5]
                contrib = result[6]
                point_error = result[7]
                prim_visible_mask = result[8]

                torch.cuda.nvtx.range_pop()  # Forward
                torch.cuda.nvtx.range_push("Losses")

                rgb = rgb + (1 - alpha[..., None]) * random_bkgd
                train_psnr = psnr(rgb, rgb_gt)
                rgb_loss = F.mse_loss(rgb, rgb_gt, reduction="none").sum(dim=-1).mean()

                ssim_loss = 1 - ssim(rgb, rgb_gt)
                w_ssim = 0.2

                # torch.cuda.nvtx.range_push("Quantile")
                # valid_depth_mask = (depth > 0).all(dim=-1)
                # quant_loss = (depth[..., 0] - depth[..., 1]).abs()
                # quant_loss = quant_loss * valid_depth_mask
                # quant_loss = quant_loss.mean()
                # w_quant = args.quantile_weight * min(2 * i / args.iterations, 1)

                # torch.cuda.nvtx.range_pop()  # Quantile
                torch.cuda.nvtx.range_push("Normal")

                normal_loss = normal_err.mean()
                # if args.use_metric3d:
                #     normal_loss += (
                #         F.mse_loss(
                #             normal[valid_depth_mask], normal_gt[valid_depth_mask]
                #         )
                #         * 1e-1
                #     )
                # else:
                #     median_depth = torch.gather(depth, dim=-1, index=mask.unsqueeze(-1))
                #     median_depth = depth_bilateral_filter(
                #         median_depth,
                #         sigma_spatial=sigma_spatial_scheduler(i),
                #         sigma_color=0.5,
                #     )
                #     est_normals = normals_from_depth(camera, median_depth.detach())
                #     normal_loss += (
                #         F.mse_loss(
                #             normal[valid_depth_mask], est_normals[valid_depth_mask]
                #         )
                #         * 1e-1
                #     )
                w_normal = normal_loss_scheduler(i)
                torch.cuda.nvtx.range_pop()  # Normal

                torch.cuda.nvtx.range_push("Contribution")
                contrib_loss = contrib.sum()
                w_contrib = contrib_loss_scheduler(i)
                torch.cuda.nvtx.range_pop()  # Contribution

                torch.cuda.nvtx.range_push("Interpenetration")
                interpenetration_loss = model.interpenetration().sum()
                w_interpenetration = interpenetration_loss_scheduler(i)
                torch.cuda.nvtx.range_pop()  # Interpenetration

                loss = (
                    rgb_loss
                    + w_ssim * ssim_loss
                    # + w_quant * quant_loss
                    + w_normal * normal_loss
                    + w_contrib * contrib_loss
                    + w_interpenetration * interpenetration_loss
                )

                torch.cuda.nvtx.range_pop()  # Losses
                torch.cuda.nvtx.range_push("Backward")

                loss.backward()

                torch.cuda.nvtx.range_pop()  # Backward
                torch.cuda.nvtx.range_push("Optimizer Step")

                model.optimizer.step()

                torch.cuda.nvtx.range_pop()  # Optimizer Step
                torch.cuda.nvtx.range_push("Stats Update")

                model.update_learning_rate(i)
                model.update_stats(contrib, point_error, prim_visible_mask)

                torch.cuda.nvtx.range_pop()  # Stats Update

                if i % 100 == 99 and not args.dry_run:
                    writer.add_scalar("train/rgb_loss", rgb_loss.item(), i)
                    # writer.add_scalar("train/quant_loss", quant_loss.item(), i)
                    writer.add_scalar("train/normal_loss", normal_loss.item(), i)
                    writer.add_scalar("train/contrib_loss", contrib_loss.item(), i)
                    writer.add_scalar(
                        "train/interpenetration_loss", interpenetration_loss.item(), i
                    )
                    writer.add_scalar("train/psnr", debiased_psnr, i)

                    num_points = model.points.shape[0]
                    writer.add_scalar("test/num_points", num_points, i)
                    test_loop(i, final=False)

                    writer.add_scalar("lr/points_lr", model.points_scheduler(i), i)
                    writer.add_scalar("lr/density_lr", model.density_scheduler(i), i)
                    writer.add_scalar("lr/radii_lr", model.radii_scheduler(i), i)
                    writer.add_scalar(
                        "lr/normals_lr", model.quaternions_scheduler(i), i
                    )
                    writer.add_scalar(
                        "lr/texel_sites_lr", model.texel_sites_scheduler(i), i
                    )
                    writer.add_scalar(
                        "lr/texel_sv_rgb_lr", model.texel_sv_rgb_scheduler(i), i
                    )
                    writer.add_scalar(
                        "lr/texel_sv_axis_lr", model.texel_sv_axis_scheduler(i), i
                    )
                    writer.add_scalar(
                        "lr/texel_height_lr", model.texel_height_scheduler(i), i
                    )

                if i % 1000 == 999 and not args.dry_run:
                    model.save_pt(f"{out_dir}/model.pt")
                    model.save_pc(f"{out_dir}/points.ply")

                torch.cuda.nvtx.range_push("Resampling")

                if i >= args.densify_from and i < args.densify_until:
                    a = (args.final_points / args.init_points) ** (
                        1 / (args.densify_until - args.densify_from - 1)
                    )
                    current_target = int(
                        args.init_points * (a ** (i - args.densify_from))
                    )
                else:
                    current_target = model.points.shape[0]

                if i < int(0.95 * args.iterations) and i % 100 == 99:
                    num_resampled = model.resample(current_target)
                    model.sort_points()
                    model.rebuild_adjacency()
                    iters_since_triangulation = 0
                else:
                    num_resampled = 0

                torch.cuda.nvtx.range_pop()  # Resampling
                torch.cuda.nvtx.range_pop()  # Train Step

                reserved_mem = torch.cuda.memory_reserved() / 1024**3
                if reserved_mem > 12.0:
                    gc.collect()
                    torch.cuda.empty_cache()

                train_psnr_ema = 0.99 * train_psnr_ema + 0.01 * train_psnr.item()
                ema_weight = 0.99 * ema_weight + 0.01
                debiased_psnr = train_psnr_ema / ema_weight

                train.set_postfix(
                    {
                        "Train PSNR": f"{debiased_psnr:.2f}",
                        "Resampled": f"{num_resampled}/{model.points.shape[0]}",
                    }
                )

        if not args.dry_run:
            model.save_pt(f"{out_dir}/model.pt")

    train_loop()
    test_loop(args.iterations, final=True)


if __name__ == "__main__":
    parser = configargparse.ArgParser()

    get_params = add_group(parser, Params)

    # Add argument to specify a custom config file
    parser.add_argument(
        "-c", "--config", is_config_file=True, help="Path to config file"
    )

    # Parse arguments
    args = parser.parse_args()

    train(get_params(args))
