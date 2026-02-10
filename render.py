import torch
import numpy as np
import numpy as np
import configargparse

from configs import *
from powerfoam.mesh_extractor import MeshExtractor

seed = 0
torch.random.manual_seed(seed)
np.random.seed(seed)

if __name__ == "__main__":
    parser = configargparse.ArgParser()
    add_group(parser, Params)

    parser.add_argument("-c", "--config",
        is_config_file=True, help="Path to config file")
    parser.add_argument("--mesh_name",
        type=str, default="mesh", help="Mesh: output mesh filename")
    parser.add_argument("--depth_trunc",
        type=float, default=-1, help="Mesh: Max depth range for TSDF")
    parser.add_argument("--voxel_size",
        type=float, default=-1, help="Mesh: voxel size for TSDF")
    parser.add_argument("--sdf_trunc",
        type=float, default=-1, help="Mesh: truncation value for TSDF")
    parser.add_argument("--mesh_res",
        type=int, default=1024, help="Mesh: resolution for mesh extraction")
    parser.add_argument("--unbounded", 
        action="store_true", help='Mesh: using unbounded mode for meshing')
    parser.add_argument("--max_cluster",
        type=int, default=50, help="Mesh: max clusters to keep when post-processing mesh")

    args = parser.parse_args()
    mesh_extractor = MeshExtractor(args, args.config)
    mesh_extractor.run()