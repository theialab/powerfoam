import os
import numpy as np
from PIL import Image
from matplotlib import cm
import configargparse
import warnings

warnings.filterwarnings("ignore")

import torch
import warp as wp

from configs import *
from data_loader import DataHandler
from powerfoam.scene import PowerfoamScene
from powerfoam.metrics import psnr, ssim_eval, lpips_eval


seed = 0
torch.random.manual_seed(seed)
np.random.seed(seed)


def test(args, config_path):
    wp.init()

    checkpoint = config_path.replace("/config.yaml", "")
    os.makedirs(os.path.join(checkpoint, "test"), exist_ok=True)

    test_data_handler = DataHandler(args)
    test_data_handler.reload("test", downsample=args.downsample[-1])

    # Setting up model
    model = PowerfoamScene(args)
    model.initialize_from_dataset(test_data_handler, device="cuda")
    model.load_pt(f"{checkpoint}/model.pt")

    def test_render():
        torch.cuda.synchronize()

        psnr_list = []
        ssim_list = []
        lpips_list = []
        with torch.no_grad():
            for i in range(len(test_data_handler.cameras)):
                camera = test_data_handler.cameras[i]
                rgb_gt = test_data_handler.rgbs[i].cuda()

                depth_quantile = 0.5 * torch.ones(
                    *rgb_gt.shape[:-1], 1, device=model.device
                )

                result = model.forward(camera, depth_quantiles=depth_quantile)
                rgb = result[0]
                normal = result[3]
                depth = result[4]

                rgb_clamped = rgb.clamp(0.0, 1.0)
                psnr_list.append(psnr(rgb_clamped, rgb_gt).item())
                ssim_list.append(ssim_eval(rgb_clamped, rgb_gt).item())
                lpips_list.append(lpips_eval(rgb_clamped, rgb_gt).item())

                # 3-panel layout (matches train.py's test_loop):
                #   (rgb_render, normal, depth_viridis)
                rgb_output = (rgb.cpu().clamp(min=0, max=1) * 255).to(torch.uint8)
                normal = ((normal + 1) * 0.5 * 255).cpu().to(torch.uint8)
                depth = depth.cpu()[:, :, 0]
                depth_min = depth[depth > 0].min()
                depth_max = depth.max()
                depth = (depth - depth_min) / (depth_max - depth_min + 1e-8)
                depth = (cm.viridis(depth.numpy())[:, :, :3] * 255).astype(np.uint8)
                depth = torch.from_numpy(depth)

                im = torch.hstack((rgb_output, normal, depth))
                im = Image.fromarray(im.numpy())
                im.save(f"{checkpoint}/test/{i:03d}.png")

        average_psnr = sum(psnr_list) / len(psnr_list)
        average_ssim = sum(ssim_list) / len(ssim_list)
        average_lpips = sum(lpips_list) / len(lpips_list)
        with open(f"{checkpoint}/metrics.txt", "w") as f:
            f.write(f"Average PSNR:  {average_psnr:.4f}\n")
            f.write(f"Average SSIM:  {average_ssim:.4f}\n")
            f.write(f"Average LPIPS: {average_lpips:.4f}\n")

        return average_psnr, average_ssim, average_lpips

    test_render()


if __name__ == "__main__":
    parser = configargparse.ArgParser()

    get_params = add_group(parser, Params)

    # Add argument to specify a custom config file
    parser.add_argument(
        "-c", "--config", is_config_file=True, help="Path to config file"
    )

    # Parse arguments
    args = parser.parse_args()

    test(get_params(args), args.config)
