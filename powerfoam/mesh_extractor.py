import os
import numpy as np
import torch
from tqdm import tqdm
import open3d as o3d

from .scene import PowerfoamScene
from .camera import TorchCamera
from data_loader import DataHandler


import warp as wp

# adapted from https://github.com/hbb1/2d-gaussian-splatting/blob/6e21151b460f0c1f4fda346df0f162b09c1b0c2f/utils/mesh_utils.py
class MeshExtractor:
    def __init__(self, args, config_path: str):
        self.args = args
        self.config_path = config_path
        self.checkpoint = config_path.replace("/config.yaml", "")
        os.makedirs(os.path.join(self.checkpoint, "test"), exist_ok=True)

        wp.init()

        # Data/model setup
        self.data = DataHandler(args)
        self.data.reload("test", downsample=args.downsample[-1])

        self.model = PowerfoamScene(args)
        self.model.initialize_from_dataset(self.data, device="cuda")
        self.model.load_pt(f"{self.checkpoint}/model.pt")
    
    @torch.no_grad()
    def render_views(self):
        render_pkg = {
            "camera": self.data.cameras,
            "rgb": [],
            "depth": [],
            "normal": [],
            "alpha": [],
        }

        torch.cuda.synchronize()
        for i in tqdm(range(len(self.data.cameras)), desc="Getting render outputs"):
            camera = self.data.cameras[i]
            rgb_gt = self.data.rgbs[i].cuda()

            depth_quantile = 0.5 * torch.ones(
                *rgb_gt.shape[:-1], 1, device=self.model.device
            )

            (rgb, alpha, normal, _, depth, *_) = self.model.forward(
                camera, depth_quantiles=depth_quantile)

            render_pkg["rgb"].append(rgb)
            render_pkg["alpha"].append(alpha)
            render_pkg["normal"].append(normal)
            render_pkg["depth"].append(depth)

        self.estimate_bounding_sphere()
        
        return render_pkg

    @torch.no_grad()
    def extract_mesh_bounded(self, outputs_pkg, voxel_size=0.002, sdf_trunc=0.1,
                            depth_trunc=4, mask_backgrond=True, min_depth=1e-4):
        """
        Perform TSDF fusion given a fixed depth range, used in the paper.
        
        voxel_size: the voxel size of the volume
        sdf_trunc: truncation value
        depth_trunc: maximum depth range, should depended on the scene's scales
        mask_backgrond: whether to mask backgroud, only works when the dataset have masks
        min_depth: depths <= min_depth will be treated as invalid (set to 0)

        return o3d.mesh
        """
        print("Running tsdf volume integration ...")
        print(f'voxel_size: {voxel_size}')
        print(f'sdf_trunc: {sdf_trunc}')
        print(f'depth_truc: {depth_trunc}')

        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length= voxel_size,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )

        for i, cam in tqdm(enumerate(outputs_pkg["camera"]),
                           desc="TSDF integration progress"):
            rgb = outputs_pkg["rgb"][i]
            depth_ray = outputs_pkg["depth"][i]
            alpha = outputs_pkg["alpha"][i]
            
            # if we have mask provided, use it
            if mask_backgrond and (alpha is not None):
                depth_ray[alpha < 0.5] = 0

            H, W = depth_ray.shape[:2]
            device = depth_ray.device
            cam = cam.to_device(device)
            # grid of pixel centers
            ii, jj = torch.meshgrid(
                torch.arange(H, device=device, dtype=torch.float32),
                torch.arange(W, device=device, dtype=torch.float32),
                indexing="ij",
            )
            # TorchCamera.get_ray_dir expects (i=x, j=y) as implemented; pass (jj, ii)
            dirs = cam.get_ray_dir(jj.reshape(-1), ii.reshape(-1))  # (H*W,3)
            dirs = torch.nn.functional.normalize(dirs, dim=-1)
            fwd = torch.cross(cam.up, cam.right)
            fwd = torch.nn.functional.normalize(fwd, dim=0)

            cos_theta = (dirs @ fwd).reshape(H, W).clamp(min=1e-8)  # (H,W)
            depth_z = depth_ray.squeeze(-1) * cos_theta             # (H,W)

            # invalidate per Open3D rules (0 ignored)
            depth_z = torch.where(torch.isfinite(depth_z), depth_z, torch.zeros_like(depth_z))
            depth_z = torch.where(depth_z > min_depth, depth_z, torch.zeros_like(depth_z))
            depth_z = torch.where(depth_z <= depth_trunc, depth_z, torch.zeros_like(depth_z))

            rgb_np = (rgb.clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
            depth_np = depth_z.cpu().numpy().astype(np.float32)
            
            rgb_np = (rgb.clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
            depth_np = depth_z.squeeze(-1).cpu().numpy().astype(np.float32)

            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(rgb_np),
                o3d.geometry.Image(depth_np),
                depth_trunc=depth_trunc,
                convert_rgb_to_intensity=False,
                depth_scale=1.0
            )
            cam_o3d = cam.to_open3d()
            volume.integrate(rgbd, intrinsic=cam_o3d.intrinsic, extrinsic=cam_o3d.extrinsic)

        mesh = volume.extract_triangle_mesh()
        return mesh
    
    @torch.no_grad()
    def extract_mesh_unbounded(self, outputs_pkg, resolution=1024):
        def contract(x):
            mag = torch.linalg.norm(x, ord=2, dim=-1)[..., None]
            return torch.where(mag < 1, x, (2 - (1 / mag)) * (x / mag))
        
        def uncontract(y):
            mag = torch.linalg.norm(y, ord=2, dim=-1)[..., None]
            return torch.where(mag < 1, y, (1 / (2-mag) * (y/mag)))

        def compute_sdf_perframe(i, points, depthmap, rgbmap, viewpoint_cam: TorchCamera):
            device = points.device
            cam = viewpoint_cam.to_device(device)

            # camera extrinsics/intrinsics
            V = cam.w2c()                         # (4,4)
            K = cam.intrinsics_matrix()           # (3,3)

            # world -> camera coordinates
            ones = torch.ones_like(points[:, :1])
            X_h = torch.cat([points, ones], dim=-1)              # (N,4)
            Xc = (V @ X_h.T)[:3].T                               # (N,3)
            z = Xc[:, 2:3]                                       # (N,1), camera-forward depth

            # project to pixel space
            fx, fy, cx, cy = K[[0, 1, 0, 1], [0, 1, 2, 2]]
            u = (fx * (Xc[:, 0] / (z[:, 0] + 1e-8))) + cx        # (N,)
            v = (fy * (Xc[:, 1] / (z[:, 0] + 1e-8))) + cy        # (N,)

            H, W = depthmap.shape[:2]
            # visibility mask in pixel domain and positive depth
            mask_proj = (
                (u >= 0.0) & (u <= (W - 1)) &
                (v >= 0.0) & (v <= (H - 1)) &
                (z[:, 0] > 0.0)
            )

            x_ndc = 2.0 * (u / (W - 1.0)) - 1.0                  # [-1,1]
            y_ndc = 2.0 * (v / (H - 1.0)) - 1.0                  # [-1,1]
            grid = torch.stack([x_ndc, y_ndc], dim=-1)           # (N,2)
            grid = grid.view(1, 1, -1, 2)                        # (1,1,N,2)

            # prepare images for sampling (NCHW)
            depth_img = depthmap.to(device).permute(2, 0, 1).unsqueeze(0)   # (1,1,H,W)
            rgb_img = rgbmap.to(device).permute(2, 0, 1).unsqueeze(0)       # (1,3,H,W)

            sampled_depth_ray = torch.nn.functional.grid_sample(
                depth_img, grid, mode='bilinear', padding_mode='border', align_corners=True
            ).view(-1, 1)                                                    # (N,1)
            sampled_rgb = torch.nn.functional.grid_sample(
                rgb_img, grid, mode='bilinear', padding_mode='border', align_corners=True
            ).view(3, -1).T                                                  # (N,3)

            # convert ray-depth to z-depth using local pixel foreshortening
            cos_theta = 1.0 / torch.sqrt(((u - cx) / (fx + 1e-8))**2 + ((v - cy) / (fy + 1e-8))**2 + 1.0)
            sampled_depth_z = sampled_depth_ray * cos_theta[:, None]
            sdf = sampled_depth_z - z
            return sdf, sampled_rgb, mask_proj

        def compute_unbounded_tsdf(samples, inv_contraction, voxel_size, return_rgb=False):
            if inv_contraction is not None:
                mask = torch.linalg.norm(samples, dim=-1) > 1
                # adaptive sdf_truncation
                sdf_trunc = 5 * voxel_size * torch.ones_like(samples[:, 0])
                sdf_trunc[mask] *= 1/(2-torch.linalg.norm(samples, dim=-1)[mask].clamp(max=1.9))
                samples = inv_contraction(samples)
            else:
                sdf_trunc = 5 * voxel_size

            tsdfs = torch.ones_like(samples[:,0]) * (-1)
            rgbs = torch.zeros((samples.shape[0], 3), device=samples.device)

            weights = torch.zeros_like(samples[:,0])
            for i in tqdm(range(len(outputs_pkg['camera'])),
                    desc="TSDF integration progress"):
                sdf, rgb, mask_proj = compute_sdf_perframe(
                    i,
                    samples,
                    depthmap=outputs_pkg['depth'][i],
                    rgbmap=outputs_pkg['rgb'][i],
                    viewpoint_cam=outputs_pkg['camera'][i],
                )

                # volume integration
                sdf = sdf.flatten()
                if return_rgb:
                    mask = mask_proj
                else:
                    mask = mask_proj & (sdf > -sdf_trunc)
                
                sdf = torch.clamp(sdf / sdf_trunc, min=-1.0, max=1.0)[mask]
                w = weights[mask]
                wp = w + 1
                tsdfs[mask] = (tsdfs[mask] * w + sdf) / wp
                rgbs[mask] = (rgbs[mask] * w[:,None] + rgb[mask]) / wp[:,None]
                
                # update weight
                weights[mask] = wp
            
            if return_rgb:
                return tsdfs, rgbs

            return tsdfs

        normalize = lambda x: (x - self.center) / self.radius
        unnormalize = lambda x: (x * self.radius) + self.center
        inv_contraction = lambda x: unnormalize(uncontract(x))

        N = resolution
        voxel_size = (self.radius * 2 / N)
        print(f"Computing sdf gird resolution {N} x {N} x {N}")
        print(f"Define the voxel_size as {voxel_size}")
        sdf_function = lambda x: compute_unbounded_tsdf(x, inv_contraction, voxel_size)
        from utils.mcubes import marching_cubes_with_contraction
        # Estimate contracted radius percentile from current scene points
        R = contract(normalize(self.model.points)).norm(dim=-1).cpu().numpy()
        R = np.quantile(R, q=0.95)
        R = min(R+0.01, 1.9)

        mesh = marching_cubes_with_contraction(
            sdf=sdf_function,
            bounding_box_min=(-R, -R, -R),
            bounding_box_max=(R, R, R),
            level=0,
            resolution=N,
            inv_contraction=inv_contraction,
        )
        
        # Convert trimesh -> open3d
        torch.cuda.empty_cache()
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices.astype(np.float64))
        o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces.astype(np.int32))

        # coloring the mesh
        print("texturing mesh ... ")
        verts = torch.tensor(np.asarray(o3d_mesh.vertices)).float().to(self.model.device)
        _, rgbs = compute_unbounded_tsdf(verts, inv_contraction=None, voxel_size=voxel_size, return_rgb=True)
        # clamp to [0,1] to ensure valid color range and avoid later 255-scaling heuristics
        rgbs = rgbs.clamp(0.0, 1.0)
        # exit(0)
        o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(rgbs.detach().cpu().numpy().astype(np.float64))

        return o3d_mesh

    def focus_point_fn(self, poses: np.ndarray) -> np.ndarray:
        """Calculate nearest point to all focal axes in poses."""
        directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
        m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
        mt_m = np.transpose(m, [0, 2, 1]) @ m
        focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
        return focus_pt
    
    def estimate_bounding_sphere(self):
        """
        Estimate the bounding sphere given camera pose
        """
        torch.cuda.empty_cache()
        c2ws = self.data.c2ws.cpu().numpy()
        poses = c2ws[:,:3,:] @ np.diag([1, -1, -1, 1])
        center = (self.focus_point_fn(poses))
        self.radius = np.linalg.norm(c2ws[:,:3,3] - center, axis=-1).min()
        self.center = torch.from_numpy(center).float().cuda()
        print(f"The estimated bounding radius is {self.radius:.2f}")
        print(f"Use at least {2.0 * self.radius:.2f} for depth_trunc")
    
    def post_process_mesh(self, mesh, cluster_to_keep=1000):
        """
        Post-process a mesh to filter out floaters and disconnected parts
        """
        import copy
        print("post processing the mesh to keep up to {} clusters".format(cluster_to_keep))
        mesh_0 = copy.deepcopy(mesh)
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
                triangle_clusters, cluster_n_triangles, cluster_area = (mesh_0.cluster_connected_triangles())

        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)
        cluster_area = np.asarray(cluster_area)
        num_clusters = len(cluster_n_triangles)
        if num_clusters == 0:
            print("post_process_mesh: no clusters found; returning original mesh")
            return mesh

        keep = min(max(int(cluster_to_keep), 1), num_clusters)
        # threshold based on the Nth largest cluster size (by triangle count)
        n_cluster = np.sort(cluster_n_triangles.copy())[-keep]
        n_cluster = max(int(n_cluster), 50)  # filter meshes smaller than 50
        triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
        mesh_0.remove_triangles_by_mask(triangles_to_remove)
        mesh_0.remove_unreferenced_vertices()
        mesh_0.remove_degenerate_triangles()
        # Normalize vertex colors to [0,1] if they came in as 0..255
        if hasattr(mesh_0, "vertex_colors") and len(mesh_0.vertex_colors) > 0:
            cols = np.asarray(mesh_0.vertex_colors)
            if cols.size > 0 and cols.max() > 1.001:
                cols = np.clip(cols / 255.0, 0.0, 1.0)
                mesh_0.vertex_colors = o3d.utility.Vector3dVector(cols)
        try:
            mesh_0.compute_vertex_normals()
        except Exception:
            pass
        print("num vertices raw {}".format(len(mesh.vertices)))
        print("num vertices post {}".format(len(mesh_0.vertices)))
        return mesh_0

    def fuse_to_mesh(self, render_pkg):
        if getattr(self.args, "unbounded", False):
            mesh = self.extract_mesh_unbounded(
                render_pkg,
                resolution=self.args.mesh_res,
            )
        else:
            depth_trunc = (2.0 * self.radius) if self.args.depth_trunc < 0 \
                                                else self.args.depth_trunc
            voxel_size = (depth_trunc / self.args.mesh_res) if self.args.voxel_size < 0 \
                                                                else self.args.voxel_size
            sdf_trunc = 5.0 * voxel_size if self.args.sdf_trunc < 0 \
                                            else self.args.sdf_trunc

            mesh = self.extract_mesh_bounded(
                render_pkg,
                voxel_size=voxel_size,
                sdf_trunc=sdf_trunc,
                depth_trunc=depth_trunc,
                mask_backgrond=True,
            )

        mesh = self.post_process_mesh(mesh, cluster_to_keep=self.args.max_cluster)
        return mesh
    
    def save_mesh(self, mesh) -> str:
        path = os.path.join(self.checkpoint, f"{self.args.mesh_name}.ply")
        o3d.io.write_triangle_mesh(
            path,
            mesh,
            write_ascii=False,
            compressed=False,
            write_vertex_normals=True,
            write_vertex_colors=True,
            print_progress=False,
        )
        return path

    def run(self):
        pkg = self.render_views()
        mesh = self.fuse_to_mesh(pkg)
        out_path = self.save_mesh(mesh)
        print(f"Mesh saved to {out_path}")