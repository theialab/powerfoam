from typing import Any
import torch
import warp as wp

from .camera import *
from .texture import *
from .rendering_math import *


TILE_WIDTH = 8
TILE_SIZE = TILE_WIDTH * TILE_WIDTH
LEAF_SIZE = 2


@wp.struct
class VisOptions:
    transmittance_threshold: float
    max_intersections: wp.int32
    depth_quantile: float
    bkgd_color: wp.vec3f


class RasterGradFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        rasterizer,
        *args,
    ):
        return rasterizer._forward(ctx, *args)

    @staticmethod
    def backward(ctx, *args):
        return ctx.rasterizer._backward(ctx, *args)


class Rasterizer:
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

        loss_accumulation = args.render_objective == "surface"
        self.loss_accumulation = loss_accumulation

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

        @wp.kernel(enable_backward=False)
        def count_visible_pinhole_kernel(
            camera: WarpCamera,
            all_spheres: wp.array(dtype=wp.vec4f),
            counts: wp.array(dtype=wp.int32),
        ):
            thread_idx = wp.tid()

            tiles_h = 1 + (camera.height - 1) // TILE_WIDTH
            # tiles_w = 1 + (camera.width - 1) // TILE_WIDTH

            assert thread_idx < all_spheres.shape[0]

            sphere = all_spheres[thread_idx]
            center = wp.vec3f(sphere[0], sphere[1], sphere[2])
            radius = sphere[3]

            v = center - camera.eye
            forward = wp.cross(camera.up, camera.right)
            if wp.length(v) < 4.0 * radius or wp.dot(v, forward) < 0.1:
                return

            uc, vc, r1, r2, vx, vy = proj_sphere_to_obb(camera, center, radius)

            # Compute AABB of OBB (for loop bounds)
            # Extent in U: |vx*r1| + |-vy*r2|
            # Extent in V: |vy*r1| + |vx*r2|
            extent_u = wp.abs(vx * r1) + wp.abs(vy * r2)
            extent_v = wp.abs(vy * r1) + wp.abs(vx * r2)

            u_min = uc - extent_u
            u_max = uc + extent_u
            v_min = vc - extent_v
            v_max = vc + extent_v

            # Map NDC to Pixel Indices
            j_min_f = (u_min + 1.0) * 0.5 * float(camera.width - 1)
            j_max_f = (u_max + 1.0) * 0.5 * float(camera.width - 1)

            i_min_f = (1.0 - v_max) * 0.5 * float(camera.height - 1)
            i_max_f = (1.0 - v_min) * 0.5 * float(camera.height - 1)

            i_min = wp.max(wp.int32(wp.floor(i_min_f)), 0)
            i_max = wp.min(wp.int32(wp.ceil(i_max_f)), camera.height - 1)
            j_min = wp.max(wp.int32(wp.floor(j_min_f)), 0)
            j_max = wp.min(wp.int32(wp.ceil(j_max_f)), camera.width - 1)

            tile_i_min = i_min // TILE_WIDTH
            tile_i_max = i_max // TILE_WIDTH
            tile_j_min = j_min // TILE_WIDTH
            tile_j_max = j_max // TILE_WIDTH

            for tile_i in range(tile_i_min, tile_i_max + 1):
                for tile_j in range(tile_j_min, tile_j_max + 1):
                    if not verify_tile_obb_intersection(
                        tile_i, tile_j, camera, TILE_WIDTH, uc, vc, r1, r2, vx, vy
                    ):
                        continue

                    tile_idx = tile_i + tile_j * tiles_h
                    wp.atomic_add(counts, tile_idx + 1, 1)

        @wp.kernel(enable_backward=False)
        def count_visible_generic_kernel(
            camera: WarpCamera,
            all_spheres: wp.array(dtype=wp.vec4f),
            cones: wp.array(dtype=wp.vec4f),
            level_offsets: wp.array(dtype=wp.int32),
            level_dims_h: wp.array(dtype=wp.int32),
            level_dims_w: wp.array(dtype=wp.int32),
            num_levels: int,
            tile_level: int,
            counts: wp.array(dtype=wp.int32),
        ):
            thread_idx = wp.tid()

            tiles_h = 1 + (camera.height - 1) // TILE_WIDTH

            assert thread_idx < all_spheres.shape[0]

            sphere = all_spheres[thread_idx]
            center = wp.vec3f(sphere[0], sphere[1], sphere[2])
            radius = sphere[3]

            v = center - camera.eye
            forward = wp.cross(camera.up, camera.right)
            if wp.length(v) < 4.0 * radius or wp.dot(v, forward) < 0.1:
                return

            stack_level = wp.mat(shape=(32, 1), dtype=wp.int32)
            stack_a = wp.mat(shape=(32, 1), dtype=wp.int32)
            stack_b = wp.mat(shape=(32, 1), dtype=wp.int32)

            stack_size = int(1)
            stack_level[0, 0] = num_levels - 1
            stack_a[0, 0] = 0
            stack_b[0, 0] = 0

            while stack_size > 0:
                stack_size -= 1
                level = stack_level[stack_size, 0]
                a = stack_a[stack_size, 0]
                b = stack_b[stack_size, 0]

                dims_w = level_dims_w[level]
                cone_idx = level_offsets[level] + a * dims_w + b
                cone = cones[cone_idx]
                cos_half_angle = cone[3]

                if cos_half_angle > 1.0:
                    continue

                cone_axis = wp.vec3f(cone[0], cone[1], cone[2])

                if sphere_cone_intersect(
                    center, radius, camera.eye, cone_axis, cos_half_angle
                ):
                    if level == tile_level:
                        tile_idx = a + b * tiles_h
                        wp.atomic_add(counts, tile_idx + 1, 1)
                    else:
                        child_level = level - 1
                        child_dims_h = level_dims_h[child_level]
                        child_dims_w = level_dims_w[child_level]
                        ca0 = 2 * a
                        ca1 = 2 * a + 1
                        cb0 = 2 * b
                        cb1 = 2 * b + 1

                        if ca0 < child_dims_h and cb0 < child_dims_w:
                            stack_level[stack_size, 0] = child_level
                            stack_a[stack_size, 0] = ca0
                            stack_b[stack_size, 0] = cb0
                            stack_size += 1

                        if ca0 < child_dims_h and cb1 < child_dims_w:
                            stack_level[stack_size, 0] = child_level
                            stack_a[stack_size, 0] = ca0
                            stack_b[stack_size, 0] = cb1
                            stack_size += 1

                        if ca1 < child_dims_h and cb0 < child_dims_w:
                            stack_level[stack_size, 0] = child_level
                            stack_a[stack_size, 0] = ca1
                            stack_b[stack_size, 0] = cb0
                            stack_size += 1

                        if ca1 < child_dims_h and cb1 < child_dims_w:
                            stack_level[stack_size, 0] = child_level
                            stack_a[stack_size, 0] = ca1
                            stack_b[stack_size, 0] = cb1
                            stack_size += 1

        @wp.kernel(enable_backward=False)
        def write_visible_pinhole_kernel(
            camera: WarpCamera,
            all_spheres: wp.array(dtype=wp.vec4f),
            offsets: wp.array(dtype=wp.int64),
            prim_indices: wp.array(dtype=wp.int32),
            sort_keys: wp.array(dtype=wp.int64),
            write_counter: wp.array(dtype=wp.int32),
        ):
            thread_idx = wp.tid()

            tiles_h = 1 + (camera.height - 1) // TILE_WIDTH
            # tiles_w = 1 + (camera.width - 1) // TILE_WIDTH

            assert thread_idx < all_spheres.shape[0]

            sphere = all_spheres[thread_idx]
            center = wp.vec3f(sphere[0], sphere[1], sphere[2])
            radius = sphere[3]

            v = center - camera.eye
            forward = wp.cross(camera.up, camera.right)
            if wp.length(v) < 4.0 * radius or wp.dot(v, forward) < 0.1:
                return

            uc, vc, r1, r2, vx, vy = proj_sphere_to_obb(camera, center, radius)

            # Compute AABB of OBB (for loop bounds)
            extent_u = wp.abs(vx * r1) + wp.abs(vy * r2)
            extent_v = wp.abs(vy * r1) + wp.abs(vx * r2)

            u_min = uc - extent_u
            u_max = uc + extent_u
            v_min = vc - extent_v
            v_max = vc + extent_v

            # Map NDC to Pixel Indices
            j_min_f = (u_min + 1.0) * 0.5 * float(camera.width - 1)
            j_max_f = (u_max + 1.0) * 0.5 * float(camera.width - 1)

            i_min_f = (1.0 - v_max) * 0.5 * float(camera.height - 1)
            i_max_f = (1.0 - v_min) * 0.5 * float(camera.height - 1)

            i_min = wp.max(wp.int32(wp.floor(i_min_f)), 0)
            i_max = wp.min(wp.int32(wp.ceil(i_max_f)), camera.height - 1)
            j_min = wp.max(wp.int32(wp.floor(j_min_f)), 0)
            j_max = wp.min(wp.int32(wp.ceil(j_max_f)), camera.width - 1)

            tile_i_min = i_min // TILE_WIDTH
            tile_i_max = i_max // TILE_WIDTH
            tile_j_min = j_min // TILE_WIDTH
            tile_j_max = j_max // TILE_WIDTH

            for tile_i in range(tile_i_min, tile_i_max + 1):
                for tile_j in range(tile_j_min, tile_j_max + 1):
                    if not verify_tile_obb_intersection(
                        tile_i, tile_j, camera, TILE_WIDTH, uc, vc, r1, r2, vx, vy
                    ):
                        continue

                    tile_idx = tile_i + tile_j * tiles_h
                    offset_start = offsets[tile_idx]
                    offset_end = offsets[tile_idx + 1]
                    offset_in_tile = wp.atomic_add(write_counter, tile_idx, 1)
                    idx = offset_start + wp.int64(offset_in_tile)
                    if idx < offset_end:
                        prim_indices[idx] = thread_idx
                        pow_dist = wp.dot(v, v) - radius * radius
                        sort_keys[idx] = (
                            wp.int64(tile_idx) << wp.int64(32)
                        ) | wp.int64(wp.cast(pow_dist, wp.uint32))

        @wp.kernel(enable_backward=False)
        def write_visible_generic_kernel(
            camera: WarpCamera,
            all_spheres: wp.array(dtype=wp.vec4f),
            cones: wp.array(dtype=wp.vec4f),
            level_offsets: wp.array(dtype=wp.int32),
            level_dims_h: wp.array(dtype=wp.int32),
            level_dims_w: wp.array(dtype=wp.int32),
            num_levels: int,
            tile_level: int,
            offsets: wp.array(dtype=wp.int64),
            prim_indices: wp.array(dtype=wp.int32),
            sort_keys: wp.array(dtype=wp.int64),
            write_counter: wp.array(dtype=wp.int32),
        ):
            thread_idx = wp.tid()

            tiles_h = 1 + (camera.height - 1) // TILE_WIDTH

            assert thread_idx < all_spheres.shape[0]

            sphere = all_spheres[thread_idx]
            center = wp.vec3f(sphere[0], sphere[1], sphere[2])
            radius = sphere[3]

            v = center - camera.eye
            forward = wp.cross(camera.up, camera.right)
            if wp.length(v) < 4.0 * radius or wp.dot(v, forward) < 0.1:
                return

            stack_level = wp.mat(shape=(32, 1), dtype=wp.int32)
            stack_a = wp.mat(shape=(32, 1), dtype=wp.int32)
            stack_b = wp.mat(shape=(32, 1), dtype=wp.int32)

            stack_size = int(1)
            stack_level[0, 0] = num_levels - 1
            stack_a[0, 0] = 0
            stack_b[0, 0] = 0

            while stack_size > 0:
                stack_size -= 1
                level = stack_level[stack_size, 0]
                a = stack_a[stack_size, 0]
                b = stack_b[stack_size, 0]

                dims_w = level_dims_w[level]
                cone_idx = level_offsets[level] + a * dims_w + b
                cone = cones[cone_idx]
                cos_half_angle = cone[3]

                if cos_half_angle > 1.0:
                    continue

                cone_axis = wp.vec3f(cone[0], cone[1], cone[2])

                if sphere_cone_intersect(
                    center, radius, camera.eye, cone_axis, cos_half_angle
                ):
                    if level == tile_level:
                        tile_idx = a + b * tiles_h
                        offset_start = offsets[tile_idx]
                        offset_end = offsets[tile_idx + 1]
                        offset_in_tile = wp.atomic_add(write_counter, tile_idx, 1)
                        idx = offset_start + wp.int64(offset_in_tile)
                        if idx < offset_end:
                            prim_indices[idx] = thread_idx
                            pow_dist = wp.dot(v, v) - radius * radius
                            sort_keys[idx] = (
                                wp.int64(tile_idx) << wp.int64(32)
                            ) | wp.int64(wp.cast(pow_dist, wp.uint32))
                    else:
                        child_level = level - 1
                        child_dims_h = level_dims_h[child_level]
                        child_dims_w = level_dims_w[child_level]
                        ca0 = 2 * a
                        ca1 = 2 * a + 1
                        cb0 = 2 * b
                        cb1 = 2 * b + 1

                        if ca0 < child_dims_h and cb0 < child_dims_w:
                            stack_level[stack_size, 0] = child_level
                            stack_a[stack_size, 0] = ca0
                            stack_b[stack_size, 0] = cb0
                            stack_size += 1

                        if ca0 < child_dims_h and cb1 < child_dims_w:
                            stack_level[stack_size, 0] = child_level
                            stack_a[stack_size, 0] = ca0
                            stack_b[stack_size, 0] = cb1
                            stack_size += 1

                        if ca1 < child_dims_h and cb0 < child_dims_w:
                            stack_level[stack_size, 0] = child_level
                            stack_a[stack_size, 0] = ca1
                            stack_b[stack_size, 0] = cb0
                            stack_size += 1

                        if ca1 < child_dims_h and cb1 < child_dims_w:
                            stack_level[stack_size, 0] = child_level
                            stack_a[stack_size, 0] = ca1
                            stack_b[stack_size, 0] = cb1
                            stack_size += 1

        @wp.kernel(enable_backward=False)
        def compute_leaf_cones_kernel(
            camera: WarpCamera,
            cones: wp.array(dtype=wp.vec4f),
            leaf_dims_w: int,
        ):
            tid = wp.tid()
            a = tid // leaf_dims_w
            b = tid % leaf_dims_w
            pix_i_start = a * LEAF_SIZE
            pix_j_start = b * LEAF_SIZE

            axis_sum = wp.vec3f(0.0, 0.0, 0.0)
            valid_count = int(0)
            rays = wp.mat(shape=(4, 3), dtype=wp.float32)

            for di in range(LEAF_SIZE):
                for dj in range(LEAF_SIZE):
                    pi = wp.min(pix_i_start + di, camera.height - 1)
                    pj = wp.min(pix_j_start + dj, camera.width - 1)
                    ray = wp.vec3f(
                        camera.ray_maps[pi, pj, 3],
                        camera.ray_maps[pi, pj, 4],
                        camera.ray_maps[pi, pj, 5],
                    )
                    ray_len = wp.length(ray)
                    if ray_len > 1.0e-6:
                        ray_n = ray / ray_len
                        rays[valid_count, 0] = ray_n[0]
                        rays[valid_count, 1] = ray_n[1]
                        rays[valid_count, 2] = ray_n[2]
                        axis_sum += ray_n
                        valid_count += 1

            if valid_count == 0:
                cones[tid] = wp.vec4f(0.0, 0.0, 0.0, 2.0)
                return

            axis = wp.normalize(axis_sum)
            cos_half = float(1.0)
            for i in range(valid_count):
                r = wp.vec3f(rays[i, 0], rays[i, 1], rays[i, 2])
                cos_half = wp.min(cos_half, wp.dot(axis, r))

            cones[tid] = wp.vec4f(axis[0], axis[1], axis[2], cos_half)

        @wp.kernel(enable_backward=False)
        def merge_cones_kernel(
            cones: wp.array(dtype=wp.vec4f),
            parent_offset: int,
            child_offset: int,
            parent_dims_w: int,
            child_dims_h: int,
            child_dims_w: int,
        ):
            tid = wp.tid()
            a = tid // parent_dims_w
            b = tid % parent_dims_w

            axis_sum = wp.vec3f(0.0, 0.0, 0.0)
            valid_count = int(0)
            child_axes = wp.mat(shape=(4, 3), dtype=wp.float32)
            child_cos_halves = wp.mat(shape=(4, 1), dtype=wp.float32)

            for di in range(2):
                for dj in range(2):
                    ca = 2 * a + di
                    cb = 2 * b + dj
                    if ca < child_dims_h and cb < child_dims_w:
                        child_idx = child_offset + ca * child_dims_w + cb
                        child_cone = cones[child_idx]
                        if child_cone[3] <= 1.0:
                            child_axis = wp.vec3f(
                                child_cone[0], child_cone[1], child_cone[2]
                            )
                            child_axes[valid_count, 0] = child_axis[0]
                            child_axes[valid_count, 1] = child_axis[1]
                            child_axes[valid_count, 2] = child_axis[2]
                            child_cos_halves[valid_count, 0] = child_cone[3]
                            axis_sum += child_axis
                            valid_count += 1

            if valid_count == 0:
                cones[parent_offset + tid] = wp.vec4f(0.0, 0.0, 0.0, 2.0)
                return

            axis = wp.normalize(axis_sum)
            cos_half = float(1.0)
            for i in range(valid_count):
                ca_vec = wp.vec3f(
                    child_axes[i, 0], child_axes[i, 1], child_axes[i, 2]
                )
                cch = child_cos_halves[i, 0]
                dot_val = wp.dot(axis, ca_vec)
                sin_angle = wp.sqrt(wp.max(1.0 - dot_val * dot_val, 0.0))
                sin_child = wp.sqrt(wp.max(1.0 - cch * cch, 0.0))
                extent = dot_val * cch - sin_angle * sin_child
                cos_half = wp.min(cos_half, extent)

            cones[parent_offset + tid] = wp.vec4f(
                axis[0], axis[1], axis[2], cos_half
            )

        if args.is_pinhole:
            self.count_visible_kernel = count_visible_pinhole_kernel
            self.write_visible_kernel = write_visible_pinhole_kernel
        else:
            self.count_visible_kernel = count_visible_generic_kernel
            self.write_visible_kernel = write_visible_generic_kernel
            self.compute_leaf_cones_kernel = compute_leaf_cones_kernel
            self.merge_cones_kernel = merge_cones_kernel

        @wp.kernel(enable_backward=False)
        def prefetch_adjacency_kernel(
            all_spheres: wp.array(dtype=wp.vec4f),
            adjacency: wp.array(dtype=wp.int32),
            adjacency_offsets: wp.array(dtype=wp.int32),
            adjacency_diff: wp.array(dtype=wp.vec4h),
            num_primitives: int,
        ):
            thread_idx = wp.tid()
            if thread_idx >= num_primitives:
                return

            offset_start = adjacency_offsets[thread_idx]
            offset_end = adjacency_offsets[thread_idx + 1]

            sphere = all_spheres[thread_idx]
            for i in range(offset_start, offset_end):
                adj_idx = adjacency[i]
                adj_sphere = all_spheres[adj_idx]

                adj_diff = wp.vec4h(
                    wp.float16(adj_sphere[0] - sphere[0]),
                    wp.float16(adj_sphere[1] - sphere[1]),
                    wp.float16(adj_sphere[2] - sphere[2]),
                    wp.float16(adj_sphere[3]),
                )
                adjacency_diff[i] = adj_diff

        self.prefetch_adjacency_kernel = prefetch_adjacency_kernel

        disable_coop_prim_load = args.disable_coop_prim_load
        disable_coop_adj_load = args.disable_coop_adj_load
        is_pinhole = args.is_pinhole

        @wp.kernel(enable_backward=False)
        def benchmark_kernel(
            camera: WarpCamera,
            all_spheres: wp.array(dtype=wp.vec4f),
            all_nsigmas: wp.array(dtype=vec4s),
            all_texel_sites: wp.array2d(dtype=vec3s),
            all_texel_rgb: wp.array2d(dtype=vec3s),
            all_texel_height: wp.array2d(dtype=scalar),
            adjacency: wp.array(dtype=wp.int32),
            adjacency_offsets: wp.array(dtype=wp.int32),
            adjacency_diff: wp.array(dtype=wp.vec4h),
            tile_prim_indices: wp.array(dtype=wp.int32),
            tile_prim_indices_offsets: wp.array(dtype=wp.int64),
            transmittance_threshold: float,
            color_out: wp.array2d(dtype=wp.vec3f),
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

            pix_i = wp.min(pix_i, camera.height - 1)
            pix_j = wp.min(pix_j, camera.width - 1)

            ray_d = get_ray_dir(camera, float(pix_i), float(pix_j))
            ray_d = ray_d / wp.length(ray_d)
            ray_o = camera.eye

            rgb = wp.vec3f(0.0, 0.0, 0.0)
            log_t = float(0.0)

            prim_offset_start = tile_prim_indices_offsets[tile_idx]
            prim_offset_end = tile_prim_indices_offsets[tile_idx + 1]
            total_prims = wp.int32(prim_offset_end - prim_offset_start)

            for intersection_idx in range(total_prims):
                current_offset = prim_offset_start + wp.int64(intersection_idx)
                prim_idx = tile_prim_indices[current_offset]

                trans = wp.exp(log_t)
                if trans < transmittance_threshold:
                    break

                sphere = all_spheres[prim_idx]
                nsigma = all_nsigmas[prim_idx]
                center = wp.vec3f(sphere[0], sphere[1], sphere[2])
                radius = sphere[3]
                prim_normal = wp.vec3f(
                    float(nsigma[0]), float(nsigma[1]), float(nsigma[2])
                )
                sigma = float(nsigma[3])

                hit, t_near, t_far = ray_sphere_intersect(ray_o, ray_d, center, radius)
                if not hit:
                    continue

                adj_offset_start = adjacency_offsets[prim_idx]
                adj_offset_end = adjacency_offsets[prim_idx + 1]
                n_adj = adj_offset_end - adj_offset_start

                for adj_idx in range(n_adj):
                    current_adj_offset = adj_offset_start + adj_idx

                    adj_diff = adjacency_diff[current_adj_offset]
                    diff = wp.vec3f(
                        float(adj_diff[0]), float(adj_diff[1]), float(adj_diff[2])
                    )
                    pm_diff = float(adj_diff[3])

                    t_face, dp = ray_pface_intersect_diff(ray_o, ray_d, diff, pm_diff)

                    t_far = wp.min(t_face, t_far) if dp >= 0.0 else t_far
                    t_near = wp.max(t_face, t_near) if dp < 0.0 else t_near

                    if t_near > t_far:
                        break

                if t_near > t_far:
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

                dt = t_far - t_near
                if hit and dt > 0.0:
                    delta_log_t = -sigma * dt
                    alpha = 1.0 - wp.exp(delta_log_t)
                    trans = wp.exp(log_t)

                    rgb += color * alpha * trans
                    log_t += delta_log_t

            if not oob:
                ray_trans = wp.exp(log_t)
                rgb += bkgd_color * ray_trans
                color_out[pix_i, pix_j] = rgb

        self.benchmark_kernel = benchmark_kernel

        @wp.kernel(enable_backward=False)
        def visualization_kernel(
            camera: WarpCamera,
            all_spheres: wp.array(dtype=wp.vec4f),
            all_nsigmas: wp.array(dtype=vec4s),
            all_texel_sites: wp.array2d(dtype=vec3s),
            all_texel_rgbh: wp.array2d(dtype=vec4s),
            all_texel_rgb: wp.array2d(dtype=vec3s),
            all_texel_height: wp.array2d(dtype=scalar),
            adjacency: wp.array(dtype=wp.int32),
            adjacency_offsets: wp.array(dtype=wp.int32),
            adjacency_diff: wp.array(dtype=wp.vec4h),
            tile_prim_indices: wp.array(dtype=wp.int32),
            tile_prim_indices_offsets: wp.array(dtype=wp.int64),
            options: VisOptions,
            color_out: wp.array2d(dtype=wp.vec3f),
            depth_out: wp.array2d(dtype=wp.float32),
            normal_out: wp.array2d(dtype=wp.vec3f),
            alpha_out: wp.array2d(dtype=wp.float32),
            intersections_out: wp.array2d(dtype=wp.int32),
        ):
            thread_idx = wp.tid()
            tile_idx = thread_idx // TILE_SIZE

            tiles_h = 1 + (camera.height - 1) // TILE_WIDTH

            tile_i = tile_idx % tiles_h
            tile_j = tile_idx // tiles_h

            idx_in_tile = thread_idx % TILE_SIZE
            i_in_tile = idx_in_tile % TILE_WIDTH
            j_in_tile = idx_in_tile // TILE_WIDTH

            pix_i = tile_i * TILE_WIDTH + i_in_tile
            pix_j = tile_j * TILE_WIDTH + j_in_tile

            oob = pix_i >= camera.height or pix_j >= camera.width

            pix_i = wp.min(pix_i, camera.height - 1)
            pix_j = wp.min(pix_j, camera.width - 1)

            ray_d = get_ray_dir(camera, float(pix_i), float(pix_j))
            ray_d = ray_d / wp.length(ray_d)
            ray_o = camera.eye

            rgb = wp.vec3f(0.0, 0.0, 0.0)
            normal_acc = wp.vec3f(0.0, 0.0, 0.0)
            depth_quantile_out = float(0.0)
            depth_quantile_found = wp.bool(False)
            log_t = float(0.0)
            n_intersections_hit = int(0)

            prim_offset_start = tile_prim_indices_offsets[tile_idx]
            prim_offset_end = tile_prim_indices_offsets[tile_idx + 1]
            total_prims = wp.int32(prim_offset_end - prim_offset_start)

            for intersection_idx in range(total_prims):
                n_intersections_hit += 1

                if intersection_idx >= options.max_intersections:
                    break

                current_offset = prim_offset_start + wp.int64(intersection_idx)
                prim_idx = tile_prim_indices[current_offset]

                trans = wp.exp(log_t)
                if trans < options.transmittance_threshold:
                    break

                sphere = all_spheres[prim_idx]
                nsigma = all_nsigmas[prim_idx]
                center = wp.vec3f(sphere[0], sphere[1], sphere[2])
                radius = sphere[3]
                prim_normal = wp.vec3f(
                    float(nsigma[0]), float(nsigma[1]), float(nsigma[2])
                )
                sigma = float(nsigma[3])

                hit, t_near, t_far = ray_sphere_intersect(ray_o, ray_d, center, radius)
                if not hit:
                    continue

                adj_offset_start = adjacency_offsets[prim_idx]
                adj_offset_end = adjacency_offsets[prim_idx + 1]
                n_adj = adj_offset_end - adj_offset_start

                for adj_idx in range(n_adj):
                    current_adj_offset = adj_offset_start + adj_idx

                    adj_diff = adjacency_diff[current_adj_offset]
                    adj_center = center + wp.vec3f(
                        float(adj_diff[0]), float(adj_diff[1]), float(adj_diff[2])
                    )
                    adj_radius = float(adj_diff[3])

                    t_face, dp = ray_pface_intersect(
                        ray_o, ray_d, center, radius, adj_center, adj_radius
                    )

                    t_far = wp.min(t_face, t_far) if dp >= 0.0 else t_far
                    t_near = wp.max(t_face, t_near) if dp < 0.0 else t_near

                    if t_near > t_far:
                        break

                if t_near > t_far:
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

                dt = t_far - t_near
                if hit and dt > 0.0:
                    delta_log_t = -sigma * dt
                    alpha = 1.0 - wp.exp(delta_log_t)
                    trans = wp.exp(log_t)

                    rgb += color * alpha * trans
                    normal_acc += prim_normal * alpha * trans
                    log_t += delta_log_t

                    if not depth_quantile_found:
                        next_trans = wp.exp(log_t)
                        if next_trans < options.depth_quantile:
                            depth_quantile_out = (
                                t_near + wp.log(trans / options.depth_quantile) / sigma
                            )
                            depth_quantile_found = wp.bool(True)

            if not oob:
                ray_trans = wp.exp(log_t)
                ray_alpha = 1.0 - ray_trans
                rgb += options.bkgd_color * ray_trans

                color_out[pix_i, pix_j] = rgb
                depth_out[pix_i, pix_j] = depth_quantile_out
                normal_out[pix_i, pix_j] = normal_acc
                alpha_out[pix_i, pix_j] = ray_alpha
                intersections_out[pix_i, pix_j] = n_intersections_hit

        self.visualization_kernel = visualization_kernel

        @wp.kernel(enable_backward=False)
        def forward_kernel(
            camera: WarpCamera,
            depth_quantiles: wp.array3d(dtype=wp.float32),
            valid_quantiles: bool,
            all_spheres: wp.array(dtype=wp.vec4f),
            all_nsigmas: wp.array(dtype=wp.vec4f),
            all_texel_sites: wp.array2d(dtype=wp.vec3f),
            all_texel_rgbh: wp.array2d(dtype=wp.vec4f),
            adjacency: wp.array(dtype=wp.int32),
            adjacency_offsets: wp.array(dtype=wp.int32),
            adjacency_diff: wp.array(dtype=wp.vec4h),
            tile_prim_indices: wp.array(dtype=wp.int32),
            tile_prim_indices_offsets: wp.array(dtype=wp.int64),
            tile_early_stop_counter: wp.array(dtype=wp.int32),
            ray_gt: wp.array2d(dtype=wp.vec3f),
            transmittance_threshold: float,
            return_point_err: bool,
            color_out: wp.array2d(dtype=wp.vec3f),
            log_t_out: wp.array2d(dtype=wp.float32),
            normal_distance_out: wp.array2d(dtype=wp.float32),
            normal_out: wp.array2d(dtype=wp.vec3f),
            quantile_depths_out: wp.array3d(dtype=wp.float32),
            err_out: wp.array2d(dtype=wp.float32),
            contrib_out: wp.array(dtype=wp.float32),
            point_err_out: wp.array(dtype=wp.float32),
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

            pix_i = wp.min(pix_i, camera.height - 1)
            pix_j = wp.min(pix_j, camera.width - 1)

            if is_pinhole:
                ray_d = get_ray_dir(camera, float(pix_i), float(pix_j))
                ray_d = ray_d / wp.length(ray_d)
                ray_o = camera.eye
            else:
                ray_o = wp.vec3f(
                    camera.ray_maps[pix_i, pix_j, 0],
                    camera.ray_maps[pix_i, pix_j, 1],
                    camera.ray_maps[pix_i, pix_j, 2],
                )
                ray_d = wp.vec3f(
                    camera.ray_maps[pix_i, pix_j, 3],
                    camera.ray_maps[pix_i, pix_j, 4],
                    camera.ray_maps[pix_i, pix_j, 5],
                )
                ray_d = ray_d / wp.length(ray_d)

            if valid_quantiles:
                ray_depth_quantiles = depth_quantiles[pix_i, pix_j]
                num_depth_quantiles = int(len(ray_depth_quantiles))
                current_quantile_idx = int(0)
                current_quantile = ray_depth_quantiles[current_quantile_idx]
            else:
                num_depth_quantiles = 0
                current_quantile_idx = 0
                current_quantile = 0.0

            early_stop = wp.bool(False)
            rgb = wp.vec3f(0.0, 0.0, 0.0)
            log_t = float(0.0)
            if loss_accumulation:
                err = float(0.0)
            if loss_accumulation or return_point_err:
                gt = ray_gt[pix_i, pix_j]
            normal_distance = float(0.0)
            normal = wp.vec3f(0.0, 0.0, 0.0)

            prim_offset_start = tile_prim_indices_offsets[tile_idx]
            prim_offset_end = tile_prim_indices_offsets[tile_idx + 1]
            total_prims = wp.int32(prim_offset_end - prim_offset_start)

            for intersection_idx in range(total_prims):
                current_offset = prim_offset_start + wp.int64(intersection_idx)
                prim_idx = tile_prim_indices[current_offset]

                trans = wp.exp(log_t)
                shared_trans = wp.tile(trans)
                max_trans = wp.tile_max(shared_trans)[0]
                if max_trans < transmittance_threshold:
                    early_stop = wp.bool(True)
                    tile_early_stop_counter[tile_idx] = intersection_idx
                    break

                if disable_coop_prim_load:
                    sphere = all_spheres[prim_idx]
                    nsigma = all_nsigmas[prim_idx]
                    center = wp.vec3f(sphere[0], sphere[1], sphere[2])
                    radius = sphere[3]
                    prim_normal = wp.vec3f(nsigma[0], nsigma[1], nsigma[2])
                    sigma = nsigma[3]

                else:
                    intersection_subidx = intersection_idx % TILE_SIZE

                    if intersection_subidx == 0:
                        local_offset = wp.min(
                            current_offset + wp.int64(idx_in_tile),
                            prim_offset_end - wp.int64(1),
                        )
                        local_prim_idx = tile_prim_indices[local_offset]
                        local_sphere = all_spheres[local_prim_idx]
                        local_nsigma = all_nsigmas[local_prim_idx]
                        shared_spheres = wp.tile(local_sphere)
                        shared_nsigmas = wp.tile(local_nsigma)

                    center = wp.vec3f(
                        shared_spheres[0, intersection_subidx],
                        shared_spheres[1, intersection_subidx],
                        shared_spheres[2, intersection_subidx],
                    )
                    radius = shared_spheres[3, intersection_subidx]
                    prim_normal = wp.vec3f(
                        shared_nsigmas[0, intersection_subidx],
                        shared_nsigmas[1, intersection_subidx],
                        shared_nsigmas[2, intersection_subidx],
                    )
                    sigma = shared_nsigmas[3, intersection_subidx]

                hit, t_near, t_far = ray_sphere_intersect(ray_o, ray_d, center, radius)
                tile_hit = wp.tile(wp.int32(hit))
                any_hit = wp.tile_sum(tile_hit)[0]
                if any_hit == 0:
                    continue

                adj_offset_start = adjacency_offsets[prim_idx]
                adj_offset_end = adjacency_offsets[prim_idx + 1]
                n_adj = adj_offset_end - adj_offset_start

                for adj_idx in range(n_adj):
                    current_adj_offset = adj_offset_start + adj_idx

                    if disable_coop_adj_load:
                        adj_diff = adjacency_diff[current_adj_offset]
                        adj_center = center + wp.vec3f(
                            float(adj_diff[0]), float(adj_diff[1]), float(adj_diff[2])
                        )
                        adj_radius = float(adj_diff[3])

                    else:
                        adj_subidx = adj_idx % TILE_SIZE

                        if adj_subidx == 0:
                            local_adj_idx = wp.min(
                                current_adj_offset + idx_in_tile,
                                adj_offset_end - 1,
                            )
                            local_adj_diff = adjacency_diff[local_adj_idx]
                            shared_adj_diff = wp.tile(local_adj_diff)

                        adj_center = center + wp.vec3f(
                            float(shared_adj_diff[0, adj_subidx]),
                            float(shared_adj_diff[1, adj_subidx]),
                            float(shared_adj_diff[2, adj_subidx]),
                        )
                        adj_radius = float(shared_adj_diff[3, adj_subidx])

                    if hit:
                        t_face, dp = ray_pface_intersect(
                            ray_o, ray_d, center, radius, adj_center, adj_radius
                        )
                        t_far = wp.min(t_face, t_far) if dp >= 0.0 else t_far
                        t_near = wp.max(t_face, t_near) if dp < 0.0 else t_near

                _, _, height, _, t_surf, dp, color, _ = plane_intersection_fwd(
                    ray_o,
                    ray_d,
                    t_near,
                    center,
                    prim_normal,
                    radius,
                    not oob and hit,
                    all_texel_sites[prim_idx],
                    all_texel_rgbh[prim_idx],
                    num_texel_sites,
                )
                t_far = wp.min(t_surf, t_far) if dp >= 0.0 else t_far
                t_near = wp.max(t_surf, t_near) if dp < 0.0 else t_near

                dt = t_far - t_near
                if hit and dt > 0.0:
                    delta_log_t = -sigma * dt
                    alpha = 1.0 - wp.exp(delta_log_t)
                    trans = wp.exp(log_t)

                    if loss_accumulation:
                        local_err = wp.length_sq(color - gt)
                        err += local_err * alpha * trans

                    rgb += color * alpha * trans
                    log_t += delta_log_t

                    contrib_out[prim_idx] += (
                        alpha * trans / float(camera.height * camera.width)
                    )
                    if return_point_err:
                        point_err_out[prim_idx] += (
                            alpha
                            * trans
                            * wp.math.norm_l1(color - gt)
                            / float(camera.height * camera.width)
                        )

                    ndv = wp.dot(prim_normal, ray_d)
                    if ndv > 0:
                        normal_distance += (ndv * ndv) * alpha * trans
                    normal += alpha * trans * prim_normal
                    next_trans = wp.exp(log_t)
                    while (
                        current_quantile_idx < num_depth_quantiles
                        and next_trans < current_quantile
                    ):
                        quantile_depth = (
                            t_near + wp.log(trans / current_quantile) / sigma
                        )
                        quantile_depths_out[pix_i, pix_j, current_quantile_idx] = (
                            quantile_depth
                        )
                        current_quantile_idx += 1
                        if current_quantile_idx < num_depth_quantiles:
                            current_quantile = ray_depth_quantiles[current_quantile_idx]

            if not oob:
                if not early_stop:
                    tile_early_stop_counter[tile_idx] = total_prims

                ray_trans = wp.exp(log_t)
                rgb += bkgd_color * ray_trans

                color_out[pix_i, pix_j] = rgb
                log_t_out[pix_i, pix_j] = log_t
                normal_distance_out[pix_i, pix_j] = normal_distance
                normal_out[pix_i, pix_j] = normal

                if loss_accumulation:
                    bkgd_err = wp.length_sq(bkgd_color - gt)
                    err += bkgd_err * ray_trans
                    err_out[pix_i, pix_j] = err

        self.forward_kernel = forward_kernel

        @wp.kernel(enable_backward=False)
        def backward_kernel(
            camera: WarpCamera,
            depth_quantiles: wp.array3d(dtype=wp.float32),
            valid_quantiles: bool,
            all_spheres: wp.array(dtype=wp.vec4f),
            all_nsigmas: wp.array(dtype=wp.vec4f),
            all_texel_sites: wp.array2d(dtype=wp.vec3f),
            all_texel_rgbh: wp.array2d(dtype=wp.vec4f),
            adjacency: wp.array(dtype=wp.int32),
            adjacency_offsets: wp.array(dtype=wp.int32),
            adjacency_diff: wp.array(dtype=wp.vec4h),
            tile_prim_indices: wp.array(dtype=wp.int32),
            tile_prim_indices_offsets: wp.array(dtype=wp.int64),
            tile_early_stop_counter: wp.array(dtype=wp.int32),
            ray_gt: wp.array2d(dtype=wp.vec3f),
            ray_color: wp.array2d(dtype=wp.vec3f),
            ray_log_t: wp.array2d(dtype=wp.float32),
            color_grad_in: wp.array2d(dtype=wp.vec3f),
            log_t_grad_in: wp.array2d(dtype=wp.float32),
            normal_distance_grad_in: wp.array2d(dtype=wp.float32),
            normal_grad_in: wp.array2d(dtype=wp.vec3f),
            quantile_depths_grad_in: wp.array3d(dtype=wp.float32),
            err_grad_in: wp.array2d(dtype=wp.float32),
            contrib_grad_in: wp.array(dtype=wp.float32),
            spheres_grad_out: wp.array(dtype=wp.vec4f),
            nsigmas_grad_out: wp.array(dtype=wp.vec4f),
            texel_sites_grad_out: wp.array2d(dtype=wp.vec3f),
            texel_rgb_grad_out: wp.array2d(dtype=wp.vec3f),
            texel_height_grad_out: wp.array2d(dtype=wp.float32),
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

            pix_i = wp.min(pix_i, camera.height - 1)
            pix_j = wp.min(pix_j, camera.width - 1)

            if is_pinhole:
                ray_d = get_ray_dir(camera, float(pix_i), float(pix_j))
                ray_d = ray_d / wp.length(ray_d)
                ray_o = camera.eye
            else:
                ray_o = wp.vec3f(
                    camera.ray_maps[pix_i, pix_j, 0],
                    camera.ray_maps[pix_i, pix_j, 1],
                    camera.ray_maps[pix_i, pix_j, 2],
                )
                ray_d = wp.vec3f(
                    camera.ray_maps[pix_i, pix_j, 3],
                    camera.ray_maps[pix_i, pix_j, 4],
                    camera.ray_maps[pix_i, pix_j, 5],
                )
                ray_d = ray_d / wp.length(ray_d)

            log_t = ray_log_t[pix_i, pix_j]
            if valid_quantiles:
                ray_depth_quantiles = depth_quantiles[pix_i, pix_j]
                num_depth_quantiles = int(len(ray_depth_quantiles))
                current_quantile_idx = int(num_depth_quantiles - 1)
                current_quantile = ray_depth_quantiles[current_quantile_idx]
            else:
                num_depth_quantiles = 0
                current_quantile_idx = -1
                current_quantile = 1.0

            ray_rgb_grad = color_grad_in[pix_i, pix_j]
            log_t_grad = log_t_grad_in[pix_i, pix_j]
            if loss_accumulation:
                ray_err_grad = err_grad_in[pix_i, pix_j]
                raise NotImplementedError()
            if valid_quantiles:
                quantile_depths_grad = quantile_depths_grad_in[pix_i, pix_j]
            normal_distance_grad = normal_distance_grad_in[pix_i, pix_j]
            normal_grad = normal_grad_in[pix_i, pix_j]

            dLdtrans = wp.dot(ray_rgb_grad, bkgd_color)
            dLdlog_t = dLdtrans * wp.exp(log_t)
            dLdlog_t += log_t_grad

            prim_offset_start = tile_prim_indices_offsets[tile_idx]
            prim_offset_end = tile_prim_indices_offsets[tile_idx + 1]
            total_prims = wp.int32(prim_offset_end - prim_offset_start)
            early_stop = tile_early_stop_counter[tile_idx]

            for intersection_idx in range(early_stop - 1, -1, -1):
                current_offset = prim_offset_start + wp.int64(intersection_idx)
                prim_idx = tile_prim_indices[current_offset]

                if disable_coop_prim_load:
                    sphere = all_spheres[prim_idx]
                    nsigma = all_nsigmas[prim_idx]
                    center = wp.vec3f(sphere[0], sphere[1], sphere[2])
                    radius = sphere[3]
                    prim_normal = wp.vec3f(nsigma[0], nsigma[1], nsigma[2])
                    sigma = nsigma[3]

                else:
                    intersection_subidx = (
                        early_stop - 1 - intersection_idx
                    ) % TILE_SIZE

                    if intersection_subidx == 0:
                        local_offset = wp.max(
                            current_offset - wp.int64(idx_in_tile),
                            wp.int64(0),
                        )
                        local_prim_idx = tile_prim_indices[local_offset]
                        local_sphere = all_spheres[local_prim_idx]
                        local_nsigma = all_nsigmas[local_prim_idx]
                        shared_sphere = wp.tile(local_sphere)
                        shared_nsigma = wp.tile(local_nsigma)

                    center = wp.vec3f(
                        shared_sphere[0, intersection_subidx],
                        shared_sphere[1, intersection_subidx],
                        shared_sphere[2, intersection_subidx],
                    )
                    radius = shared_sphere[3, intersection_subidx]
                    prim_normal = wp.vec3f(
                        shared_nsigma[0, intersection_subidx],
                        shared_nsigma[1, intersection_subidx],
                        shared_nsigma[2, intersection_subidx],
                    )
                    sigma = shared_nsigma[3, intersection_subidx]

                hit, t_near, t_far = ray_sphere_intersect(ray_o, ray_d, center, radius)
                tile_hit = wp.tile(wp.int32(hit))
                any_hit = wp.tile_sum(tile_hit)[0]
                if any_hit == 0:
                    continue

                # id => -2 none, -1 sphere, 0...n-1 face, n plane
                t_near_id = -2 if t_near < 1e-8 else -1
                t_far_id = -2 if t_far < 1e-8 else -1

                adj_offset_start = adjacency_offsets[prim_idx]
                adj_offset_end = adjacency_offsets[prim_idx + 1]
                n_adj = adj_offset_end - adj_offset_start

                for adj_idx in range(n_adj):
                    current_adj_offset = adj_offset_start + adj_idx

                    if disable_coop_adj_load:
                        adj_diff = adjacency_diff[current_adj_offset]
                        adj_center = center + wp.vec3f(
                            float(adj_diff[0]), float(adj_diff[1]), float(adj_diff[2])
                        )
                        adj_radius = float(adj_diff[3])

                    else:
                        adj_subidx = adj_idx % TILE_SIZE

                        if adj_subidx == 0:
                            local_adj_idx = wp.min(
                                current_adj_offset + idx_in_tile,
                                adj_offset_end - 1,
                            )
                            local_adj_diff = adjacency_diff[local_adj_idx]
                            shared_adj_diff = wp.tile(local_adj_diff)

                        adj_center = center + wp.vec3f(
                            float(shared_adj_diff[0, adj_subidx]),
                            float(shared_adj_diff[1, adj_subidx]),
                            float(shared_adj_diff[2, adj_subidx]),
                        )
                        adj_radius = float(shared_adj_diff[3, adj_subidx])

                    if hit:
                        t_face, dp = ray_pface_intersect(
                            ray_o, ray_d, center, radius, adj_center, adj_radius
                        )
                        t_far_id = (
                            adj_idx if (dp >= 0.0 and t_face < t_far) else t_far_id
                        )
                        t_far = wp.min(t_face, t_far) if dp >= 0.0 else t_far
                        t_near_id = (
                            adj_idx if (dp < 0.0 and t_face > t_near) else t_near_id
                        )
                        t_near = wp.max(t_face, t_near) if dp < 0.0 else t_near

                _t_near, _t_near_id = t_near, t_near_id
                _t_far, _t_far_id = t_far, t_far_id
                _t_surf, _dp, height, _wsum, t_surf, dp, color, wsum = (
                    plane_intersection_fwd(
                        ray_o,
                        ray_d,
                        t_near,
                        center,
                        prim_normal,
                        radius,
                        not oob and hit,
                        all_texel_sites[prim_idx],
                        all_texel_rgbh[prim_idx],
                        num_texel_sites,
                    )
                )
                t_far_id = n_adj if (dp >= 0.0 and t_surf < t_far) else t_far_id
                t_far = wp.min(t_surf, t_far) if dp >= 0.0 else t_far
                t_near_id = n_adj if (dp < 0.0 and t_surf > t_near) else t_near_id
                t_near = wp.max(t_surf, t_near) if dp < 0.0 else t_near

                dt = t_far - t_near
                dLdt_near, dLdt_far = 0.0, 0.0
                dLdcenter, dLdradius = wp.vec3f(0.0, 0.0, 0.0), 0.0
                dLdnormal, dLdsigma = wp.vec3f(0.0, 0.0, 0.0), 0.0
                dLdcolor = wp.vec3f(0.0, 0.0, 0.0)
                if not oob and hit and dt > 0.0:
                    delta_log_t = -sigma * dt
                    log_t -= delta_log_t
                    alpha = 1.0 - wp.exp(delta_log_t)
                    trans = wp.exp(log_t)
                    ndv = wp.dot(prim_normal, ray_d)

                    dLdlog_t_i = dLdlog_t
                    dLddelta_log_t = dLdlog_t

                    dLdalpha = wp.dot(ray_rgb_grad, color) * trans
                    if ndv > 0:
                        dLdalpha += normal_distance_grad * ndv * trans
                    dLdalpha += (
                        contrib_grad_in[prim_idx]
                        * trans
                        / float(camera.height * camera.width)
                    )
                    dLdalpha += wp.dot(normal_grad, prim_normal) * trans

                    dLdtrans = wp.dot(ray_rgb_grad, color) * alpha
                    if ndv > 0:
                        dLdtrans += normal_distance_grad * ndv * alpha
                    dLdtrans += (
                        contrib_grad_in[prim_idx]
                        * alpha
                        / float(camera.height * camera.width)
                    )
                    dLdtrans += wp.dot(normal_grad, prim_normal) * alpha

                    dLdcolor += ray_rgb_grad * trans * alpha
                    dLdlog_t_i += dLdtrans * wp.exp(log_t)
                    dLddelta_log_t += -dLdalpha * wp.exp(delta_log_t)

                    dLdsigma += dLddelta_log_t * -dt
                    dLdt_near += dLddelta_log_t * sigma
                    dLdt_far += dLddelta_log_t * -sigma

                    while current_quantile_idx >= 0 and trans > current_quantile:
                        dLdquantile_depth = quantile_depths_grad[current_quantile_idx]
                        dLdsigma += (
                            -dLdquantile_depth
                            * wp.log(trans / current_quantile)
                            / (sigma * sigma + 1e-6)
                        )
                        dLdt_near += dLdquantile_depth
                        dLdlog_t_i += dLdquantile_depth / (sigma + 1e-6)
                        current_quantile_idx -= 1
                        if current_quantile_idx >= 0:
                            current_quantile = ray_depth_quantiles[current_quantile_idx]

                    dLdnormal += alpha * trans * normal_grad
                    if ndv > 0:
                        dLdnormal += normal_distance_grad * alpha * trans * ray_d

                    dLdlog_t = dLdlog_t_i

                if t_near_id == n_adj:
                    dLdt_surf = dLdt_near
                elif t_far_id == n_adj:
                    dLdt_surf = dLdt_far
                else:
                    dLdt_surf = 0.0
                _dLdcenter, _dLdnormal, dLd_t_near = plane_intersection_bwd(
                    ray_o,
                    ray_d,
                    _t_near,
                    center,
                    prim_normal,
                    radius,
                    not oob and hit and dt > 0.0,
                    all_texel_sites[prim_idx],
                    all_texel_rgbh[prim_idx],
                    _dp,
                    _t_surf,
                    height,
                    _wsum,
                    dp,
                    t_surf,
                    color,
                    wsum,
                    dLdcolor,
                    dLdt_surf,
                    num_texel_sites,
                    texel_sites_grad_out[prim_idx],
                    texel_rgb_grad_out[prim_idx],
                    texel_height_grad_out[prim_idx],
                )
                dLdcenter += _dLdcenter
                dLdnormal += _dLdnormal
                if t_near_id != n_adj:
                    dLd_t_near += dLdt_near

                if not oob and hit and dt > 0.0:
                    # Process both t_near and t_far without duplicating code
                    for i in range(2):
                        if i == 0:
                            t_id, dLdt = _t_near_id, dLd_t_near
                        else:
                            t_id, dLdt = _t_far_id, dLdt_far

                        if t_id == -1:  # sphere intersection
                            dt_neardc, dt_fardc, dt_neardr, dt_fardr = (
                                ray_sphere_intersect_bwd(ray_o, ray_d, center, radius)
                            )
                            if i == 0:
                                dLdcenter += dLdt * dt_neardc
                                dLdradius += dLdt * dt_neardr
                            else:
                                dLdcenter += dLdt * dt_fardc
                                dLdradius += dLdt * dt_fardr
                        elif t_id >= 0 and t_id < n_adj:  # power face intersection
                            adjacent = adjacency[adj_offset_start + t_id]
                            adj_sphere = all_spheres[adjacent]
                            adj_center = wp.vec3f(
                                adj_sphere[0], adj_sphere[1], adj_sphere[2]
                            )
                            adj_radius = adj_sphere[3]
                            dtdcenter, dtdradius, dtdadjcenter, dtdadjradius = (
                                ray_pface_intersect_bwd(
                                    ray_o, ray_d, center, radius, adj_center, adj_radius
                                )
                            )
                            dLdcenter += dLdt * dtdcenter
                            dLdradius += dLdt * dtdradius
                            dLdadjcenter = dLdt * dtdadjcenter
                            dLdadjradius = dLdt * dtdadjradius
                            if not oob:
                                dLdadjsphere = wp.vec4f(
                                    dLdadjcenter[0],
                                    dLdadjcenter[1],
                                    dLdadjcenter[2],
                                    dLdadjradius,
                                )
                                spheres_grad_out[adjacent] += dLdadjsphere

                    dLdsphere = wp.vec4f(
                        dLdcenter[0], dLdcenter[1], dLdcenter[2], dLdradius
                    )
                    spheres_grad_out[prim_idx] += dLdsphere
                    dLdnsigma = wp.vec4f(
                        dLdnormal[0], dLdnormal[1], dLdnormal[2], dLdsigma
                    )
                    nsigmas_grad_out[prim_idx] += dLdnsigma

        self.backward_kernel = backward_kernel

    def _precompute_cones(self, camera, tiles_h, tiles_w):
        leaf_dims_h = (camera.height + LEAF_SIZE - 1) // LEAF_SIZE
        leaf_dims_w = (camera.width + LEAF_SIZE - 1) // LEAF_SIZE

        level_offsets_list = []
        level_dims_h_list = []
        level_dims_w_list = []
        total_nodes = 0

        dh, dw = leaf_dims_h, leaf_dims_w
        while True:
            level_offsets_list.append(total_nodes)
            level_dims_h_list.append(dh)
            level_dims_w_list.append(dw)
            total_nodes += dh * dw
            if dh == 1 and dw == 1:
                break
            dh = (dh + 1) // 2
            dw = (dw + 1) // 2

        num_levels = len(level_offsets_list)

        tile_level = 0
        sz = LEAF_SIZE
        while sz < TILE_WIDTH:
            tile_level += 1
            sz *= 2

        cones = torch.zeros(total_nodes, 4, dtype=torch.float32, device=self.device)
        level_offsets = torch.tensor(
            level_offsets_list, dtype=torch.int32, device=self.device
        )
        level_dims_h = torch.tensor(
            level_dims_h_list, dtype=torch.int32, device=self.device
        )
        level_dims_w = torch.tensor(
            level_dims_w_list, dtype=torch.int32, device=self.device
        )

        warp_camera = camera.to_warp()

        wp.launch(
            self.compute_leaf_cones_kernel,
            dim=leaf_dims_h * leaf_dims_w,
            inputs=[warp_camera, cones, leaf_dims_w],
            block_dim=min(256, leaf_dims_h * leaf_dims_w),
        )

        for l in range(1, num_levels):
            pdh = level_dims_h_list[l]
            pdw = level_dims_w_list[l]
            cdh = level_dims_h_list[l - 1]
            cdw = level_dims_w_list[l - 1]
            wp.launch(
                self.merge_cones_kernel,
                dim=pdh * pdw,
                inputs=[
                    cones,
                    level_offsets_list[l],
                    level_offsets_list[l - 1],
                    pdw,
                    cdh,
                    cdw,
                ],
                block_dim=min(256, pdh * pdw),
            )

        return cones, level_offsets, level_dims_h, level_dims_w, num_levels, tile_level

    def _forward(
        self,
        grad_ctx,
        camera,
        depth_quantiles,
        points,
        radii,
        density,
        normals,
        texel_sites,
        texel_rgb,
        texel_height,
        adjacency,
        adjacency_offsets,
        ray_gt,
        return_point_err,
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

            if self.attr_dtype != "float":
                raise ValueError("Forward pass only supports float precision")

            all_spheres = torch.cat([points, radii[:, None]], dim=-1)
            all_nsigmas = torch.cat([normals, density[:, None]], dim=-1)
            all_texel_sites = texel_sites
            all_texel_rgbh = torch.cat(
                [texel_rgb, texel_height[..., None]], dim=-1
            )

            if ray_gt is None and self.args.render_objective == "surface":
                raise ValueError(
                    "Ground truth rays must be provided for surface loss accumulation"
                )
            if ray_gt is None and return_point_err:
                raise ValueError(
                    "Ground truth rays must be provided to return point error"
                )

            color_out = torch.zeros(
                (camera.height, camera.width, 3),
                dtype=torch.float32,
                device=self.device,
            )
            log_t_out = torch.zeros(
                (camera.height, camera.width), dtype=torch.float32, device=self.device
            )
            normal_distance_out = torch.zeros(
                (camera.height, camera.width), dtype=torch.float32, device=self.device
            )
            normal_out = torch.zeros(
                (camera.height, camera.width, 3),
                dtype=torch.float32,
                device=self.device,
            )
            contrib_out = torch.zeros(
                num_points, dtype=torch.float32, device=self.device
            )
            if depth_quantiles is not None:
                valid_quantiles = True
                quantile_depths_out = -1 * torch.ones(
                    *depth_quantiles.shape, dtype=torch.float32, device=self.device
                )
            else:
                valid_quantiles = False
                quantile_depths_out = None
            if return_point_err:
                point_err_out = torch.zeros(
                    num_points, dtype=torch.float32, device=self.device
                )
            else:
                point_err_out = None
            if self.loss_accumulation:
                err_out = torch.zeros(
                    (camera.height, camera.width),
                    dtype=torch.float32,
                    device=self.device,
                )
            else:
                err_out = None

            tile_inter_counts = torch.zeros(
                total_tiles + 1, dtype=torch.int32, device=self.device
            )

            if self.args.is_pinhole:
                wp.launch(
                    self.count_visible_kernel,
                    dim=num_points,
                    inputs=[
                        camera.to_warp(),
                        all_spheres.detach(),
                        tile_inter_counts,
                    ],
                    block_dim=256,
                )
            else:
                cones, level_offsets, level_dims_h, level_dims_w, num_levels, tile_level = (
                    self._precompute_cones(camera, tiles_h, tiles_w)
                )
                wp.launch(
                    self.count_visible_kernel,
                    dim=num_points,
                    inputs=[
                        camera.to_warp(),
                        all_spheres.detach(),
                        cones,
                        level_offsets,
                        level_dims_h,
                        level_dims_w,
                        num_levels,
                        tile_level,
                        tile_inter_counts,
                    ],
                    block_dim=256,
                )

            offsets = torch.cumsum(tile_inter_counts.long(), dim=0)
            n_intersections = offsets[-1].item()

            tile_prim_indices = torch.zeros(
                n_intersections, dtype=torch.int32, device=self.device
            )
            sort_keys = torch.zeros(
                n_intersections, dtype=torch.int64, device=self.device
            )
            write_counter = torch.zeros(
                total_tiles, dtype=torch.int32, device=self.device
            )

            if self.args.is_pinhole:
                wp.launch(
                    self.write_visible_kernel,
                    dim=num_points,
                    inputs=[
                        camera.to_warp(),
                        all_spheres.detach(),
                        offsets,
                        tile_prim_indices,
                        sort_keys,
                        write_counter,
                    ],
                    block_dim=256,
                )
            else:
                wp.launch(
                    self.write_visible_kernel,
                    dim=num_points,
                    inputs=[
                        camera.to_warp(),
                        all_spheres.detach(),
                        cones,
                        level_offsets,
                        level_dims_h,
                        level_dims_w,
                        num_levels,
                        tile_level,
                        offsets,
                        tile_prim_indices,
                        sort_keys,
                        write_counter,
                    ],
                    block_dim=256,
                )

            tile_prim_indices_long = tile_prim_indices.long()
            prim_visible_mask = torch.zeros(
                num_points, dtype=torch.bool, device=self.device
            )
            prim_visible_mask[tile_prim_indices_long] = True

            tile_prim_perm = torch.argsort(sort_keys)
            tile_prim_indices = tile_prim_indices[tile_prim_perm]

            tile_early_stop_counter = torch.zeros(
                total_tiles, dtype=torch.int32, device=self.device
            )

            adjacency_diff = torch.zeros(
                adjacency.shape[0], 4, dtype=torch.float16, device=self.device
            )

            wp.launch(
                self.prefetch_adjacency_kernel,
                dim=num_points,
                inputs=[
                    all_spheres.detach(),
                    adjacency,
                    adjacency_offsets,
                    adjacency_diff,
                    num_points,
                ],
                block_dim=256,
            )

            grad_ctx.rasterizer = self
            grad_ctx.camera = camera

            raster_threads = total_tiles * TILE_SIZE
            wp.launch(
                self.forward_kernel,
                dim=raster_threads,
                inputs=[
                    camera.to_warp(),
                    depth_quantiles,
                    valid_quantiles,
                    all_spheres.detach(),
                    all_nsigmas.detach(),
                    all_texel_sites.detach(),
                    all_texel_rgbh.detach(),
                    adjacency,
                    adjacency_offsets,
                    adjacency_diff,
                    tile_prim_indices,
                    offsets,
                    tile_early_stop_counter,
                    ray_gt,
                    transmittance_threshold,
                    return_point_err,
                    color_out,
                    log_t_out,
                    normal_distance_out,
                    normal_out,
                    quantile_depths_out,
                    err_out,
                    contrib_out,
                    point_err_out,
                ],
                block_dim=TILE_SIZE,
            )

            grad_ctx.save_for_backward(
                depth_quantiles,
                all_spheres,
                all_nsigmas,
                all_texel_sites,
                all_texel_rgbh,
                adjacency,
                adjacency_offsets,
                adjacency_diff,
                tile_prim_indices,
                offsets,
                tile_early_stop_counter,
                ray_gt,
                color_out,
                log_t_out,
            )

            opacity_out = 1.0 - torch.exp(log_t_out)

        return (
            color_out,
            opacity_out,
            normal_distance_out,
            normal_out,
            quantile_depths_out,
            err_out,
            contrib_out,
            point_err_out,
            prim_visible_mask,
        )

    def forward(self, *args):
        return RasterGradFn.apply(self, *args)

    def _backward(
        self,
        grad_ctx,
        grad_color_in,
        grad_opacity_in,
        grad_normal_distance_in,
        grad_normal_in,
        grad_quantile_depths_in,
        grad_err_in,
        grad_contrib_in,
        grad_point_err_in,
        grad_prim_visible_mask_in,
    ):
        del grad_point_err_in, grad_prim_visible_mask_in  # Unused
        with wp.ScopedDevice(str(self.device)):
            torch_stream = torch.cuda.current_stream()
            wp_stream = wp.stream_from_torch(torch_stream)
            wp.set_stream(wp_stream)

            tiles_h = 1 + (grad_ctx.camera.height - 1) // TILE_WIDTH
            tiles_w = 1 + (grad_ctx.camera.width - 1) // TILE_WIDTH
            total_tiles = tiles_h * tiles_w

            if self.attr_dtype != "float":
                raise ValueError("Backward pass only supports float precision")

            (
                depth_quantiles,
                all_spheres,
                all_nsigmas,
                all_texel_sites,
                all_texel_rgbh,
                adjacency,
                adjacency_offsets,
                adjacency_diff,
                tile_prim_indices,
                offsets,
                tile_early_stop_counter,
                ray_gt,
                color_out,
                log_t_out,
            ) = grad_ctx.saved_tensors

            valid_quantiles = depth_quantiles is not None
            grad_log_t_in = -grad_opacity_in * torch.exp(log_t_out)

            spheres_grad_out = torch.zeros_like(all_spheres)
            nsigmas_grad_out = torch.zeros_like(all_nsigmas)
            texel_sites_grad_out = torch.zeros_like(all_texel_sites)
            texel_rgb_grad_out = torch.zeros_like(all_texel_rgbh[..., :3])
            texel_height_grad_out = torch.zeros_like(all_texel_rgbh[..., 3])

            raster_threads = total_tiles * TILE_SIZE
            wp.launch(
                self.backward_kernel,
                dim=raster_threads,
                inputs=[
                    grad_ctx.camera.to_warp(),
                    depth_quantiles,
                    valid_quantiles,
                    all_spheres.detach(),
                    all_nsigmas.detach(),
                    all_texel_sites.detach(),
                    all_texel_rgbh.detach(),
                    adjacency,
                    adjacency_offsets,
                    adjacency_diff,
                    tile_prim_indices,
                    offsets,
                    tile_early_stop_counter,
                    ray_gt.detach() if ray_gt is not None else None,
                    color_out.detach(),
                    log_t_out.detach(),
                    grad_color_in.detach(),
                    grad_log_t_in.detach(),
                    grad_normal_distance_in.detach(),
                    grad_normal_in.detach(),
                    (
                        grad_quantile_depths_in.detach()
                        if grad_quantile_depths_in is not None
                        else None
                    ),
                    grad_err_in.detach() if grad_err_in is not None else None,
                    grad_contrib_in.detach(),
                    spheres_grad_out,
                    nsigmas_grad_out,
                    texel_sites_grad_out,
                    texel_rgb_grad_out,
                    texel_height_grad_out,
                ],
                block_dim=TILE_SIZE,
            )

        spheres_grad_out[~spheres_grad_out.isfinite()] = 0
        nsigmas_grad_out[~nsigmas_grad_out.isfinite()] = 0
        texel_sites_grad_out[~texel_sites_grad_out.isfinite()] = 0
        texel_rgb_grad_out[~texel_rgb_grad_out.isfinite()] = 0
        texel_height_grad_out[~texel_height_grad_out.isfinite()] = 0
        # assert not torch.isnan(spheres_grad_out).any(), "spheres_grad_out has NaNs"
        # assert not torch.isnan(nsigmas_grad_out).any(), "nsigmas_grad_out has NaNs"
        # assert not torch.isnan(texel_rgb_grad_out).any(), "texel_rgb_grad_out has NaNs"

        del grad_ctx.rasterizer, grad_ctx.camera

        return (
            None,  # Rasterizer
            None,  # camera
            None,  # depth_quantiles
            spheres_grad_out[:, :3],  # points
            spheres_grad_out[:, 3],  # radii
            nsigmas_grad_out[:, 3],  # density
            nsigmas_grad_out[:, :3],  # normals
            texel_sites_grad_out,  # texel_sites
            texel_rgb_grad_out,  # rgb_from_sv
            texel_height_grad_out,  # height_from_sv
            None,  # adjacency
            None,  # adjacency_offsets
            None,  # ray_gt
            None,  # return_point_err
        )

    def benchmark(
        self,
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

            color_out = torch.zeros(
                (camera.height, camera.width, 3),
                dtype=torch.float32,
                device=self.device,
            )

            tile_inter_counts = torch.zeros(
                total_tiles + 1, dtype=torch.int32, device=self.device
            )

            if self.args.is_pinhole:
                wp.launch(
                    self.count_visible_kernel,
                    dim=num_points,
                    inputs=[
                        camera.to_warp(),
                        all_spheres.detach(),
                        tile_inter_counts,
                    ],
                    block_dim=256,
                )
            else:
                cones, level_offsets, level_dims_h, level_dims_w, num_levels, tile_level = (
                    self._precompute_cones(camera, tiles_h, tiles_w)
                )
                wp.launch(
                    self.count_visible_kernel,
                    dim=num_points,
                    inputs=[
                        camera.to_warp(),
                        all_spheres.detach(),
                        cones,
                        level_offsets,
                        level_dims_h,
                        level_dims_w,
                        num_levels,
                        tile_level,
                        tile_inter_counts,
                    ],
                    block_dim=256,
                )

            offsets = torch.cumsum(tile_inter_counts.long(), dim=0)
            n_intersections = offsets[-1].item()

            tile_prim_indices = torch.zeros(
                n_intersections, dtype=torch.int32, device=self.device
            )
            sort_keys = torch.zeros(
                n_intersections, dtype=torch.int64, device=self.device
            )
            write_counter = torch.zeros(
                total_tiles, dtype=torch.int32, device=self.device
            )

            if self.args.is_pinhole:
                wp.launch(
                    self.write_visible_kernel,
                    dim=num_points,
                    inputs=[
                        camera.to_warp(),
                        all_spheres.detach(),
                        offsets,
                        tile_prim_indices,
                        sort_keys,
                        write_counter,
                    ],
                    block_dim=256,
                )
            else:
                wp.launch(
                    self.write_visible_kernel,
                    dim=num_points,
                    inputs=[
                        camera.to_warp(),
                        all_spheres.detach(),
                        cones,
                        level_offsets,
                        level_dims_h,
                        level_dims_w,
                        num_levels,
                        tile_level,
                        offsets,
                        tile_prim_indices,
                        sort_keys,
                        write_counter,
                    ],
                    block_dim=256,
                )

            tile_prim_perm = torch.argsort(sort_keys)
            tile_prim_indices = tile_prim_indices[tile_prim_perm]

            raster_threads = total_tiles * TILE_SIZE
            wp.launch(
                self.benchmark_kernel,
                dim=raster_threads,
                inputs=[
                    camera.to_warp(),
                    all_spheres.detach(),
                    all_nsigmas.detach(),
                    all_texel_sites.detach(),
                    all_texel_rgb.detach(),
                    all_texel_height.detach(),
                    adjacency,
                    adjacency_offsets,
                    adjacency_diff,
                    tile_prim_indices,
                    offsets,
                    transmittance_threshold,
                    color_out,
                ],
                block_dim=TILE_SIZE,
            )

            return color_out

    def visualize(
        self,
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
        vis_options=None,
    ):
        if vis_options is None:
            vis_options = VisOptions()
            vis_options.transmittance_threshold = 1e-3
            vis_options.max_intersections = 1024
            vis_options.bkgd_color = wp.vec3f(0.0, 0.0, 0.0)
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
            depth_out = torch.zeros(
                (camera.height, camera.width),
                dtype=torch.float32,
                device=self.device,
            )
            normal_out = torch.zeros(
                (camera.height, camera.width, 3),
                dtype=torch.float32,
                device=self.device,
            )
            alpha_out = torch.zeros(
                (camera.height, camera.width),
                dtype=torch.float32,
                device=self.device,
            )
            intersections_out = torch.zeros(
                (camera.height, camera.width),
                dtype=torch.int32,
                device=self.device,
            )

            tile_inter_counts = torch.zeros(
                total_tiles + 1, dtype=torch.int32, device=self.device
            )

            wp.launch(
                self.count_visible_kernel,
                dim=num_points,
                inputs=[camera.to_warp(), all_spheres.detach(), tile_inter_counts],
                block_dim=256,
            )

            offsets = torch.cumsum(tile_inter_counts.long(), dim=0)
            n_intersections = offsets[-1].item()

            tile_prim_indices = torch.zeros(
                n_intersections, dtype=torch.int32, device=self.device
            )
            sort_keys = torch.zeros(
                n_intersections, dtype=torch.int64, device=self.device
            )
            write_counter = torch.zeros(
                total_tiles, dtype=torch.int32, device=self.device
            )

            wp.launch(
                self.write_visible_kernel,
                dim=num_points,
                inputs=[
                    camera.to_warp(),
                    all_spheres.detach(),
                    offsets,
                    tile_prim_indices,
                    sort_keys,
                    write_counter,
                ],
                block_dim=256,
            )

            tile_prim_perm = torch.argsort(sort_keys)
            tile_prim_indices = tile_prim_indices[tile_prim_perm]

            adjacency_diff = torch.zeros(
                adjacency.shape[0], 4, dtype=torch.float16, device=self.device
            )

            wp.launch(
                self.prefetch_adjacency_kernel,
                dim=num_points,
                inputs=[
                    all_spheres.detach(),
                    adjacency,
                    adjacency_offsets,
                    adjacency_diff,
                    num_points,
                ],
                block_dim=256,
            )

            raster_threads = total_tiles * TILE_SIZE
            wp.launch(
                self.visualization_kernel,
                dim=raster_threads,
                inputs=[
                    camera.to_warp(),
                    all_spheres.detach(),
                    all_nsigmas.detach(),
                    all_texel_sites.detach(),
                    all_texel_rgbh.detach(),
                    all_texel_rgb.detach(),
                    all_texel_height.detach(),
                    adjacency,
                    adjacency_offsets,
                    adjacency_diff,
                    tile_prim_indices,
                    offsets,
                    vis_options,
                    color_out,
                    depth_out,
                    normal_out,
                    alpha_out,
                    intersections_out,
                ],
                block_dim=TILE_SIZE,
            )

            return color_out, depth_out, normal_out, alpha_out, intersections_out
