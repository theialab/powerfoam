import os
import numpy as np
from PIL import Image
import configargparse
import warnings

warnings.filterwarnings("ignore")

import torch
import warp as wp

from configs import *
from data_loader import DataHandler
from powerfoam.scene import PowerfoamScene
from powerfoam.geometry import normals_from_depth
from powerfoam.metrics import psnr


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
        with torch.no_grad():
            for i in range(len(test_data_handler.cameras)):
                camera = test_data_handler.cameras[i]
                rgb_gt = test_data_handler.rgbs[i].cuda()

                result = model.forward(camera)
                rgb = result[0]

                img_psnr = psnr(rgb, rgb_gt)
                psnr_list.append(img_psnr.item())

                error = (rgb - rgb_gt).abs()
                error = np.uint8(error.cpu() * 255)
                rgb_output = np.uint8(rgb.cpu().clamp(min=0, max=1) * 255)
                rgb_gt = np.uint8(rgb_gt.cpu() * 255)

                im = Image.fromarray(np.hstack((rgb_gt, rgb_output, error)))
                im.save(f"{checkpoint}/test/rgb_{i:03d}_{img_psnr:.2f}.png")

        average_psnr = sum(psnr_list) / len(psnr_list)
        f = open(f"{checkpoint}/metrics.txt", "w")
        f.write(f"Average PSNR: {average_psnr}")
        f.close()

        return average_psnr

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
