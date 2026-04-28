import torch
import warp as wp

from .warp_interop import launch_kernel_from_torch


@wp.kernel(enable_backward=False)
def count_adjacent(
    bvh_id: wp.uint64,
    min: wp.array(dtype=wp.vec3f),
    max: wp.array(dtype=wp.vec3f),
    counts: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    min_i = min[i]
    max_i = max[i]
    c = 0.5 * (min_i + max_i)
    r = 0.5 * (max_i.x - min_i.x)

    query = wp.bvh_query_aabb(bvh_id, min_i, max_i)
    j = wp.int32(0)

    while wp.bvh_query_next(query, j):
        if j == i:
            continue

        min_j = min[j]
        max_j = max[j]
        c_j = 0.5 * (min_j + max_j)
        r_j = 0.5 * (max_j.x - min_j.x)
        d = wp.length(c - c_j)
        if d < (r + r_j):
            wp.atomic_add(counts, i + 1, 1)


@wp.kernel(enable_backward=False)
def write_adjacent(
    bvh_id: wp.uint64,
    min: wp.array(dtype=wp.vec3f),
    max: wp.array(dtype=wp.vec3f),
    offsets: wp.array(dtype=wp.int32),
    adjacent: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    min_i = min[i]
    max_i = max[i]
    c = 0.5 * (min_i + max_i)
    r = 0.5 * (max_i.x - min_i.x)

    offset_start = offsets[i]
    offset_end = offsets[i + 1]
    idx = offset_start

    query = wp.bvh_query_aabb(bvh_id, min_i, max_i)
    j = wp.int32(0)

    while wp.bvh_query_next(query, j):
        if j == i:
            continue

        min_j = min[j]
        max_j = max[j]
        c_j = 0.5 * (min_j + max_j)
        r_j = 0.5 * (max_j.x - min_j.x)
        d = wp.length(c - c_j)
        if d < (r + r_j):
            if idx < offset_end:
                adjacent[idx] = j
                idx += 1


class AABBTree:
    def __init__(self, device):
        self.device = device

    def update(self, centers, radii):
        self.min = wp.from_torch(
            centers - radii[:, None], dtype=wp.vec3f, requires_grad=False
        )
        self.max = wp.from_torch(
            centers + radii[:, None], dtype=wp.vec3f, requires_grad=False
        )

        torch_stream = torch.cuda.current_stream()
        wp_stream = wp.stream_from_torch(torch_stream)

        with wp.ScopedDevice(str(self.device)):
            with wp.ScopedStream(wp_stream):
                if hasattr(self, "tree"):
                    del self.tree
                self.tree = wp.Bvh(
                    lowers=self.min,
                    uppers=self.max,
                )

    def build_cech_complex(self):
        counts = torch.zeros(
            self.min.shape[0] + 1, dtype=torch.int32, device=self.device
        )

        launch_kernel_from_torch(
            device=self.device,
            kernel=count_adjacent,
            dim=(self.min.shape[0],),
            inputs=[self.tree.id, self.min, self.max, counts],
            block_dim=256,
        )

        offsets = torch.cumsum(counts, dim=0).to(torch.int32)
        max_offset = offsets[-1].item()
        if max_offset > 2**29:
            print(max_offset)
        adjacent = torch.zeros(max_offset, dtype=torch.int32, device=self.device)

        launch_kernel_from_torch(
            device=self.device,
            kernel=write_adjacent,
            dim=(self.min.shape[0],),
            inputs=[self.tree.id, self.min, self.max, offsets, adjacent],
            block_dim=256,
        )

        return adjacent, offsets
