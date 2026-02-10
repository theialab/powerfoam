from typing import Any
import torch
import warp as wp

from .camera import *
from .texture import *
from .rendering_math import *
from .sh_utils import *


TILE_WIDTH = 8
TILE_SIZE = TILE_WIDTH * TILE_WIDTH


class RayTracer:
    def __init__(self, args, device, attr_dtype="float"):
        self.device = device
        self.args = args

        if attr_dtype == "float":
            scalar = wp.float32
            vec3s = wp.vec3f
            vec4s = wp.vec4f
            self.tscalar = torch.float32
        elif attr_dtype == "half":
            scalar = wp.float16
            vec3s = wp.vec3h
            vec4s = wp.vec4h
            self.tscalar = torch.float16
        else:
            raise ValueError(f"Unsupported attribute dtype: {attr_dtype}")
        self.attr_dtype = attr_dtype

        bkgd_color = wp.vec3f(0.0, 0.0, 0.0)
        num_texel_sites = args.num_texel_sites

        # wp.config.mode = "debug"

        temp = wp.constant(10.0)
        max_sites = wp.constant(8)

        @wp.func
        def plane_intersection_fwd_local(
            ray_origin: wp.vec3f,
            ray_direction: wp.vec3f,
            t_near: float,
            plane_origin: wp.vec3f,
            plane_normal: wp.vec3f,
            radius: float,
            sites: wp.array(dtype=vec3s),
            rgbs: wp.array(dtype=vec3s),
            heights: wp.array(dtype=scalar),
            num_sites: int,
        ):
            _t_surf, _dp = ray_plane_intersect(
                ray_origin, ray_direction, plane_origin, plane_normal
            )
            _t_query = t_near if _dp >= 0.0 else wp.max(t_near, _t_surf)
            _intersection_pt = ray_origin + _t_query * ray_direction

            inv_radius_sq = 1.0 / (radius * radius)

            height_sum = float(0.0)
            _weight_sum = float(0.0)
            for i in range(num_sites):
                site = sites[i]
                site_f = wp.vec3f(float(site[0]), float(site[1]), float(site[2]))
                dist_sq = wp.length_sq(_intersection_pt - site_f) * inv_radius_sq
                weight = wp.exp(-temp * dist_sq)

                height = float(heights[i])
                height_sum += weight * height
                _weight_sum += weight

            _weight_sum = wp.max(_weight_sum, 1e-20)
            height_out = height_sum / _weight_sum

            t_surf, dp = ray_plane_intersect(
                ray_origin,
                ray_direction,
                plane_origin,
                plane_normal,
                wp.float32(height_out),
            )
            t_query = t_near if dp >= 0.0 else wp.max(t_near, t_surf)
            intersection_pt = ray_origin + t_query * ray_direction

            rgb_sum = wp.vec3f(0.0, 0.0, 0.0)
            weight_sum = float(0.0)
            for i in range(num_sites):
                site = sites[i]
                site_f = wp.vec3f(float(site[0]), float(site[1]), float(site[2]))
                dist_sq = wp.length_sq(intersection_pt - site_f) * inv_radius_sq
                weight = wp.exp(-temp * dist_sq)

                _rgb = rgbs[i]
                rgb = wp.vec3f(float(_rgb[0]), float(_rgb[1]), float(_rgb[2]))
                rgb_sum += weight * rgb
                weight_sum += weight

            weight_sum = wp.max(weight_sum, 1e-20)
            rgb_out = rgb_sum / weight_sum

            return (
                _t_surf,
                _dp,
                height_out,
                _weight_sum,
                t_surf,
                dp,
                rgb_out,
                weight_sum,
            )

        @wp.kernel
        def benchmark_kernel(
            camera: WarpCamera,
            start_point_idx: int,
            all_spheres: wp.array(dtype=wp.vec4f),
            all_nsigmas: wp.array(dtype=vec4s),
            all_texel_sites: wp.array2d(dtype=vec3s),
            all_texel_rgbh: wp.array2d(dtype=vec4s),
            all_texel_rgb: wp.array2d(dtype=vec3s),
            all_texel_height: wp.array2d(dtype=scalar),
            adjacency: wp.array(dtype=wp.int32),
            adjacency_offsets: wp.array(dtype=wp.int32),
            adjacency_diff: wp.array(dtype=wp.vec4h),
            transmittance_threshold: float,
            color_out: wp.array2d(dtype=wp.vec3f),
            median_depth_out: wp.array2d(dtype=wp.float32),
            normal_out: wp.array2d(dtype=wp.vec3f),
        ):
            thread_idx = wp.tid()
            tile_idx = thread_idx // TILE_SIZE

            tiles_h = 1 + (camera.height - 1) // TILE_WIDTH
            # tiles_w = 1 + (camera.width - 1) // TILE_WIDTH

            tile_i = tile_idx % tiles_h
            tile_j = tile_idx // tiles_h

            idx_in_tile = thread_idx % TILE_SIZE
            i_in_tile = idx_in_tile % TILE_WIDTH
            j_in_tile = idx_in_tile // TILE_WIDTH

            pix_i = tile_i * TILE_WIDTH + i_in_tile
            pix_j = tile_j * TILE_WIDTH + j_in_tile

            oob = pix_i >= camera.height or pix_j >= camera.width
            if oob:
                return

            pix_i = wp.min(pix_i, camera.height - 1)
            pix_j = wp.min(pix_j, camera.width - 1)

            ray_d = get_ray_dir(camera, float(pix_i), float(pix_j))
            ray_d = ray_d / wp.length(ray_d)
            ray_o = camera.eye

            rgb = wp.vec3f(0.0, 0.0, 0.0)
            log_t = float(0.0)
            # median_depth = float(0.0)
            # normal = wp.vec3f(0.0, 0.0, 0.0)

            prim_idx = start_point_idx
            pt_near = float(0.0)

            while True:
                trans = wp.exp(log_t)
                if trans < transmittance_threshold or prim_idx == int(0x7FFFFFFF):
                    break

                sphere = all_spheres[prim_idx]
                center = wp.vec3f(sphere[0], sphere[1], sphere[2])
                radius = sphere[3]

                hit, t_near, t_far = ray_sphere_intersect(ray_o, ray_d, center, radius)
                v = center - camera.eye
                if wp.length(v) < 4.0 * radius:
                    hit = False

                adj_offset_start = adjacency_offsets[prim_idx]
                adj_offset_end = adjacency_offsets[prim_idx + 1]
                n_adj = adj_offset_end - adj_offset_start

                next_prim_idx = int(0x7FFFFFFF)
                pt_far = float(1e10)
                for adj_idx in range(n_adj):
                    current_adj_offset = adj_offset_start + adj_idx
                    adj_point_idx = adjacency[current_adj_offset]
                    adj_diff = adjacency_diff[current_adj_offset]

                    diff = wp.vec3f(
                        float(adj_diff[0]), float(adj_diff[1]), float(adj_diff[2])
                    )
                    pm_diff = float(adj_diff[3])

                    t_face, dp = ray_pface_intersect_diff(ray_o, ray_d, diff, pm_diff)

                    if dp >= 0.0 and t_face < pt_far:  # and t_face > pt_near
                        next_prim_idx = int(adj_point_idx)
                        pt_far = t_face

                    t_far = wp.min(t_face, t_far) if dp >= 0.0 else t_far
                    t_near = wp.max(t_face, t_near) if dp < 0.0 else t_near

                nsigma = all_nsigmas[prim_idx]
                prim_normal = wp.vec3f(
                    float(nsigma[0]), float(nsigma[1]), float(nsigma[2])
                )
                sigma = float(nsigma[3])

                if not hit or t_near > t_far or sigma < 1e-3:
                    prim_idx = next_prim_idx
                    pt_near = wp.max(pt_near, pt_far)
                    continue

                _, _, height, _, t_surf, dp, color, _ = plane_intersection_fwd_local(
                    ray_o,
                    ray_d,
                    t_near,
                    center,
                    prim_normal,
                    radius,
                    all_texel_sites[prim_idx],
                    all_texel_rgb[prim_idx],
                    all_texel_height[prim_idx],
                    num_texel_sites,
                )
                t_far = wp.min(t_surf, t_far) if dp >= 0.0 else t_far
                t_near = wp.max(t_surf, t_near) if dp < 0.0 else t_near

                prim_idx = next_prim_idx
                pt_near = wp.max(pt_near, pt_far)
                dt = t_far - t_near
                if hit and dt > 0.0:
                    delta_log_t = -sigma * dt
                    alpha = 1.0 - wp.exp(delta_log_t)

                    rgb += color * alpha * trans
                    # normal += alpha * trans * prim_normal
                    log_t += delta_log_t
                    # next_trans = wp.exp(log_t)
                    # if next_trans < 0.5 and trans >= 0.5:
                    #     median_depth = t_near + wp.log(trans / 0.5) / sigma

            ray_trans = wp.exp(log_t)
            rgb += bkgd_color * ray_trans

            color_out[pix_i, pix_j] = rgb
            # normal_out[pix_i, pix_j] = normal
            # median_depth_out[pix_i, pix_j] = median_depth

        self.benchmark_kernel = benchmark_kernel

    def benchmark(
        self,
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
        transmittance_threshold=1e-3,
    ):
        with wp.ScopedDevice(str(self.device)):
            torch_stream = torch.cuda.current_stream()
            wp_stream = wp.stream_from_torch(torch_stream)
            wp.set_stream(wp_stream)

            num_points = points.shape[0]

            tiles_h = 1 + (camera.height - 1) // TILE_WIDTH
            tiles_w = 1 + (camera.width - 1) // TILE_WIDTH
            total_tiles = tiles_h * tiles_w

            all_spheres = torch.cat([points, radii[:, None]], dim=-1).to(torch.float32)
            all_nsigmas = torch.cat([normals, density[:, None]], dim=-1).to(
                self.tscalar
            )
            all_texel_sites = texel_sites.to(self.tscalar)
            all_texel_rgb = texel_rgb.to(self.tscalar)
            all_texel_height = texel_height.to(self.tscalar)
            all_texel_rgbh = torch.cat(
                [all_texel_rgb, all_texel_height[..., None]], dim=-1
            )

            color_out = torch.zeros(
                (camera.height, camera.width, 3),
                dtype=torch.float32,
                device=self.device,
            )
            median_depth_out = torch.zeros(
                (camera.height, camera.width),
                dtype=torch.float32,
                device=self.device,
            )
            normal_out = torch.zeros(
                (camera.height, camera.width, 3),
                dtype=torch.float32,
                device=self.device,
            )

            ray_trace_threads = total_tiles * TILE_SIZE
            wp.launch(
                self.benchmark_kernel,
                dim=ray_trace_threads,
                inputs=[
                    camera.to_warp(),
                    start_point_idx,
                    all_spheres.detach(),
                    all_nsigmas.detach(),
                    all_texel_sites.detach(),
                    all_texel_rgbh.detach(),
                    all_texel_rgb.detach(),
                    all_texel_height.detach(),
                    adjacency,
                    adjacency_offsets,
                    adjacency_diff,
                    transmittance_threshold,
                    color_out,
                    median_depth_out,
                    normal_out,
                ],
                block_dim=TILE_SIZE,
            )

            return color_out, normal_out, median_depth_out
