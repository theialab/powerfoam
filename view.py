"""Standalone viewer for trained PowerFoam checkpoints.

Usage:
    python view.py -c output/bonsai/config.yaml
"""

import configargparse
import warp as wp
import torch

from configs import Params, add_group
from data_loader import DataHandler
from powerfoam.scene import PowerfoamScene
from powerfoam.viewer import Viewer


def view(args, config_path):
    wp.init()

    checkpoint = config_path.replace("/config.yaml", "")

    # Load dataset (needed for SphericalVoronoi initialization)
    data_handler = DataHandler(args)
    data_handler.reload("train", downsample=args.downsample[-1])
    print(f"Loaded dataset: {args.scene}")

    # Initialize and load model
    model = PowerfoamScene(args)
    model.initialize_from_dataset(data_handler, device="cuda")
    model.load_pt(f"{checkpoint}/model.pt")
    model.declare_optimizers(args, args.iterations)
    model.sort_points()
    model.update_vis_cache()
    print(f"Loaded checkpoint: {checkpoint} ({model.points.shape[0]} points)")

    # Launch viewer (no training); uses rasterizer.
    viewer = Viewer(model, data_handler.cameras[0], world_up=data_handler.viewer_up)
    viewer.run()


if __name__ == "__main__":
    parser = configargparse.ArgParser()

    get_params = add_group(parser, Params)

    # Add argument to specify a custom config file
    parser.add_argument(
        "-c", "--config", is_config_file=True, help="Path to config file"
    )

    # Parse arguments
    args = parser.parse_args()

    view(get_params(args), args.config)
