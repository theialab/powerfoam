from typing import Any
import torch
import torch.nn.functional as F
import warp as wp


@wp.func
def expand_bits(v: wp.uint64):
    v = v & wp.uint64(0x00000000001FFFFF)
    v = (v | (v << wp.uint64(32))) & wp.uint64(0x001F00000000FFFF)
    v = (v | (v << wp.uint64(16))) & wp.uint64(0x001F0000FF0000FF)
    v = (v | (v << wp.uint64(8))) & wp.uint64(0x100F00F00F00F00F)
    v = (v | (v << wp.uint64(4))) & wp.uint64(0x10C30C30C30C30C3)
    v = (v | (v << wp.uint64(2))) & wp.uint64(0x1249249249249249)
    return v


@wp.kernel
def morton_code_kernel(
    points: wp.array(dtype=wp.vec3f),
    lower: wp.vec3f,
    upper: wp.vec3f,
    codes: wp.array(dtype=wp.int64),
):
    tid = wp.tid()
    p = points[tid]

    # Normalize to [0, 1]
    dims = upper - lower
    # Avoid division by zero
    dims = wp.vec3f(wp.max(dims[0], 1e-6), wp.max(dims[1], 1e-6), wp.max(dims[2], 1e-6))

    normalized = wp.cw_div(p - lower, dims)

    # Quantize to 21 bits [0, 2097151]
    resolution = float(2097152.0)

    x = wp.min(wp.max(normalized[0] * resolution, 0.0), resolution - 1.0)
    y = wp.min(wp.max(normalized[1] * resolution, 0.0), resolution - 1.0)
    z = wp.min(wp.max(normalized[2] * resolution, 0.0), resolution - 1.0)

    xx = expand_bits(wp.uint64(x))
    yy = expand_bits(wp.uint64(y))
    zz = expand_bits(wp.uint64(z))

    code = xx * wp.uint64(4) + yy * wp.uint64(2) + zz
    codes[tid] = wp.int64(code)


def morton_sort(points: torch.Tensor):
    """
    Sort points using Morton codes (Z-order curve) and Radix sort.
    Returns the sorted indices.
    """
    if points.shape[0] == 0:
        return torch.zeros(0, dtype=torch.int32, device=points.device)

    # Compute bounding box
    min_p = points.min(dim=0)[0]
    max_p = points.max(dim=0)[0]

    # Warp inputs
    device = points.device
    num_points = points.shape[0]

    with wp.ScopedDevice(str(device)):
        wp_points = wp.from_torch(points, dtype=wp.vec3f)
        indices = torch.arange(num_points, dtype=torch.int32, device=device)

        # Warp radix sort requires 2*N buffer
        wp_indices = wp.zeros(2 * num_points, dtype=wp.int32)
        wp.copy(wp_indices, wp.from_torch(indices), count=num_points)

        wp_codes = wp.zeros(2 * num_points, dtype=wp.int64)
        lower = wp.vec3f(float(min_p[0]), float(min_p[1]), float(min_p[2]))
        upper = wp.vec3f(float(max_p[0]), float(max_p[1]), float(max_p[2]))

        wp.launch(
            kernel=morton_code_kernel,
            dim=num_points,
            inputs=[wp_points, lower, upper, wp_codes],
        )

        wp.utils.radix_sort_pairs(wp_codes, wp_indices, num_points)
        return wp.to_torch(wp_indices)[:num_points]


@wp.kernel(enable_backward=True)
def interpenetration_kernel(
    spheres: wp.array(dtype=wp.vec4f),
    adjacency: wp.array(dtype=wp.int32),
    adjacency_offsets: wp.array(dtype=wp.int32),
    areas: wp.array(dtype=wp.float32),
):
    thread_idx = wp.tid()
    sphere = spheres[thread_idx]
    center = wp.vec3f(sphere[0], sphere[1], sphere[2])
    radius = sphere[3]

    prim_offset_start = adjacency_offsets[thread_idx]
    prim_offset_end = adjacency_offsets[thread_idx + 1]
    total_prims = wp.int32(prim_offset_end - prim_offset_start)

    for intersection_idx in range(total_prims):
        current_offset = prim_offset_start + intersection_idx
        neighbor_idx = adjacency[current_offset]

        neighbor_sphere = spheres[neighbor_idx]
        neighbor_center = wp.vec3f(
            neighbor_sphere[0], neighbor_sphere[1], neighbor_sphere[2]
        )
        neighbor_radius = neighbor_sphere[3]

        d = wp.length(center - neighbor_center)

        if d < (radius + neighbor_radius):
            areas[thread_idx] += (radius + neighbor_radius - d) ** 2.0


class InterpenetrationFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        points,
        radii,
        adjacency,
        adjacency_offsets,
    ):
        with wp.ScopedDevice(str(points.device)):
            num_points = points.shape[0]

            spheres = torch.cat([points, radii[:, None]], dim=-1).to(torch.float32)

            ctx.spheres = wp.from_torch(spheres, dtype=wp.vec4f, requires_grad=True)
            ctx.adjacency = adjacency
            ctx.adjacency_offsets = adjacency_offsets
            ctx.areas = wp.zeros(
                num_points,
                dtype=wp.float32,
                device=str(points.device),
                requires_grad=True,
            )

            wp.launch(
                kernel=interpenetration_kernel,
                dim=num_points,
                inputs=[
                    ctx.spheres,
                    ctx.adjacency,
                    ctx.adjacency_offsets,
                ],
                outputs=[ctx.areas],
                block_dim=128,
            )

            return wp.to_torch(ctx.areas)

    @staticmethod
    def backward(ctx, areas_grad_in):
        with wp.ScopedDevice(str(ctx.spheres.device)):

            areas_grad_in = areas_grad_in.contiguous()
            ctx.areas.grad = wp.from_torch(areas_grad_in, dtype=wp.float32)
            num_points = ctx.spheres.shape[0]

            wp.launch(
                kernel=interpenetration_kernel,
                dim=num_points,
                inputs=[
                    ctx.spheres,
                    ctx.adjacency,
                    ctx.adjacency_offsets,
                ],
                outputs=[ctx.areas],
                adj_inputs=[
                    ctx.spheres.grad,
                    None,
                    None,
                ],
                adj_outputs=[
                    ctx.areas.grad,
                ],
                block_dim=128,
                adjoint=True,
            )

            spheres_grad_out = wp.to_torch(ctx.spheres.grad)

            return spheres_grad_out[..., :3], spheres_grad_out[..., 3], None, None


@wp.kernel(enable_backward=False)
def bilateral_filter_kernel(
    image: wp.array2d(dtype=wp.float32),
    filtered_image: wp.array2d(dtype=wp.float32),
    width: int,
    height: int,
    window: int,
    sigma_spatial: float,
    sigma_color: float,
):
    thread_idx = wp.tid()
    pix_i = thread_idx % width
    pix_j = thread_idx // width

    if pix_i >= width or pix_j >= height:
        return

    color = image[pix_j, pix_i]
    filtered_color = float(0.0)
    denom = float(0.0)

    for i in range(window):
        for j in range(window):
            offset_i = i - window // 2
            offset_j = j - window // 2

            neighbor_i = pix_i + offset_i
            neighbor_j = pix_j + offset_j

            if (
                neighbor_i < 0
                or neighbor_i >= width
                or neighbor_j < 0
                or neighbor_j >= height
            ):
                continue

            color_dist = image[neighbor_j, neighbor_i] - color
            color_dist2 = color_dist * color_dist
            color_weight = wp.exp(-color_dist2 / (2.0 * sigma_color * sigma_color))

            spatial_dist2 = float(offset_i * offset_i + offset_j * offset_j)
            spatial_weight = wp.exp(
                -spatial_dist2 / (2.0 * sigma_spatial * sigma_spatial)
            )

            weight = spatial_weight * color_weight

            filtered_color += weight * image[neighbor_j, neighbor_i]
            denom += weight

    filtered_image[pix_j, pix_i] = filtered_color / denom


def depth_bilateral_filter(
    image: torch.Tensor,
    window: int = 7,
    sigma_spatial: float = 3.0,
    sigma_color: float = 0.1,
) -> torch.Tensor:
    height, width, _ = image.shape
    filtered_image = torch.zeros_like(image)

    num_threads = width * height
    wp.launch(
        kernel=bilateral_filter_kernel,
        dim=num_threads,
        inputs=[
            image.detach(),
            filtered_image.detach(),
            width,
            height,
            window,
            sigma_spatial,
            sigma_color,
        ],
        block_dim=256,
    )

    return filtered_image


def normals_from_depth(camera, depth):
    # Compute normals from depth map using finite differences
    device = depth.device

    eye = camera.eye.cuda(non_blocking=True)
    up, right = camera.up.cuda(non_blocking=True), camera.right.cuda(non_blocking=True)
    forward = torch.cross(up, right, dim=-1)
    forward = forward / torch.norm(forward, dim=-1, keepdim=True)

    valid_mask = depth > 0

    i, j = torch.meshgrid(
        torch.arange(camera.height, device=device),
        torch.arange(camera.width, device=device),
    )
    x = 2.0 * j / (float(camera.width) - 1.0) - 1.0
    y = 1.0 - 2.0 * i / (float(camera.height) - 1.0)
    ray_dir = (
        x[:, :, None] * right[None, None, :]
        + y[:, :, None] * up[None, None, :]
        + forward[None, None, :]
    )
    ray_dir = ray_dir / torch.norm(ray_dir, dim=-1, keepdim=True)
    points = eye[None, None, :] + depth * ray_dir

    # Compute normals using finite differences
    dx = torch.zeros_like(points)
    dy = torch.zeros_like(points)
    dx[:, 1:-1, :] = (points[:, 2:, :] - points[:, :-2, :]) / 2.0
    dy[1:-1, :, :] = (points[2:, :, :] - points[:-2, :, :]) / 2.0

    # Compute normal as cross product of gradients
    normals = torch.cross(dx, dy, dim=-1)
    normals = normals / (torch.norm(normals, dim=-1, keepdim=True) + 1e-8)

    # Ensure normals point towards the camera (flip if necessary)
    # Compute vector from point to camera
    to_camera = eye[None, None, :] - points
    to_camera = to_camera / (torch.norm(to_camera, dim=-1, keepdim=True) + 1e-8)

    # Flip normal if it points away from camera
    dot_product = torch.sum(normals * to_camera, dim=-1, keepdim=True)
    normals = torch.where(dot_product < 0, -normals, normals)

    # Set normals to zero where depth is invalid
    normals = normals * valid_mask.float()

    return normals
