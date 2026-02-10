from dataclasses import dataclass

import torch
import warp as wp
import open3d as o3d


@wp.struct
class WarpCamera:
    eye: wp.vec3f
    right: wp.vec3f
    up: wp.vec3f
    width: int
    height: int


@wp.func
def get_ray_dir(camera: WarpCamera, i: float, j: float) -> wp.vec3f:
    forward = wp.normalize(wp.cross(camera.up, camera.right))
    x = 2.0 * j / (float(camera.width) - 1.0) - 1.0
    y = 1.0 - 2.0 * i / (float(camera.height) - 1.0)
    return x * camera.right + y * camera.up + forward


@wp.func
def proj_sphere_to_circle(camera: WarpCamera, center: wp.vec3f, radius: float):
    C = center - camera.eye
    C2 = wp.length_sq(C)
    R = radius
    R2 = R * R
    X = camera.right
    Y = camera.up
    Z = wp.normalize(wp.cross(Y, X))

    a = wp.dot(C, X) ** 2.0 - (C2 - R2) * wp.dot(X, X)
    b = wp.dot(C, Y) ** 2.0 - (C2 - R2) * wp.dot(Y, Y)
    f = wp.dot(C, Z) ** 2.0 - (C2 - R2) * wp.dot(Z, Z)

    c = 2.0 * wp.dot(C, X) * wp.dot(C, Y)
    d = 2.0 * wp.dot(C, X) * wp.dot(C, Z)
    e = 2.0 * wp.dot(C, Y) * wp.dot(C, Z)

    x0 = (c * e - 2.0 * b * d) / (4.0 * a * b - c * c)
    y0 = (c * d - 2.0 * a * e) / (4.0 * a * b - c * c)

    F0 = a * x0 * x0 + b * y0 * y0 + c * x0 * y0 + d * x0 + e * y0 + f
    l_min = (a + b + wp.sqrt((a - b) * (a - b) + c * c)) / 2.0
    R_circ = wp.sqrt(-F0 / l_min)

    i0 = (1.0 - y0) * 0.5 * float(camera.height - 1)
    j0 = (x0 + 1.0) * 0.5 * float(camera.width - 1)
    R_pix = R_circ * 0.5 * float(camera.width - 1)

    return i0, j0, R_pix


@wp.func
def proj_sphere_to_obb(camera: WarpCamera, center: wp.vec3f, radius: float):
    # Returns center(u,v), half_width (major), half_height (minor), and axis angle (cos, sin)
    C = center - camera.eye
    C2 = wp.length_sq(C)
    R = radius
    R2 = R * R
    X = camera.right
    Y = camera.up
    Z = wp.normalize(wp.cross(Y, X))

    a = wp.dot(C, X) ** 2.0 - (C2 - R2) * wp.dot(X, X)
    b = wp.dot(C, Y) ** 2.0 - (C2 - R2) * wp.dot(Y, Y)
    f = wp.dot(C, Z) ** 2.0 - (C2 - R2) * wp.dot(Z, Z)

    c = 2.0 * wp.dot(C, X) * wp.dot(C, Y)
    d = 2.0 * wp.dot(C, X) * wp.dot(C, Z)
    e = 2.0 * wp.dot(C, Y) * wp.dot(C, Z)

    denom = 4.0 * a * b - c * c
    if wp.abs(denom) < 1e-8:
        return 0.0, 0.0, 0.0, 0.0, 1.0, 0.0

    uc = (c * e - 2.0 * b * d) / denom
    vc = (c * d - 2.0 * a * e) / denom

    # Value at center F0 (negative? based on a,b < 0)
    F0 = a * uc * uc + b * vc * vc + c * uc * vc + d * uc + e * vc + f

    term_sqrt = wp.sqrt((a - b) * (a - b) + c * c)
    l1 = (
        a + b + term_sqrt
    ) / 2.0  # Magnitude smaller (since a+b negative) -> Major Axis
    l2 = (a + b - term_sqrt) / 2.0  # Magnitude larger -> Minor Axis

    r1 = wp.sqrt(-F0 / l1)
    r2 = wp.sqrt(-F0 / l2)

    vx = 0.5 * c
    vy = l1 - a

    norm = wp.sqrt(vx * vx + vy * vy)
    if norm < 1e-6:
        vx = 1.0
        vy = 0.0
    else:
        vx = vx / norm
        vy = vy / norm

    return uc, vc, r1, r2, vx, vy


@wp.func
def verify_tile_obb_intersection(
    tile_i: int,
    tile_j: int,
    camera: WarpCamera,
    TILE_WIDTH: int,
    uc: float,
    vc: float,
    r1: float,
    r2: float,
    vx: float,
    vy: float,
):
    # Tile corners
    u_min = 2.0 * float(tile_j * TILE_WIDTH) / float(camera.width - 1) - 1.0
    u_max = 2.0 * float((tile_j + 1) * TILE_WIDTH) / float(camera.width - 1) - 1.0
    v_max = 1.0 - 2.0 * float(tile_i * TILE_WIDTH) / float(camera.height - 1)
    v_min = 1.0 - 2.0 * float((tile_i + 1) * TILE_WIDTH) / float(camera.height - 1)

    # Tile center and half-extents
    u_center = (u_min + u_max) * 0.5
    v_center = (v_min + v_max) * 0.5
    eu = (u_max - u_min) * 0.5
    ev = (v_max - v_min) * 0.5

    # Difference vector
    du = u_center - uc
    dv = v_center - vc

    # Separating Axis Theorem (SAT)
    # Axis 1: OBB Major Axis (vx, vy)
    # Project dist: dot(d, axis)
    # Project tile radius: eu * |axis.u| + ev * |axis.v|
    # Check: |proj_dist| > r1 + proj_tile_radius

    # Axis (vx, vy)
    proj_dist_1 = du * vx + dv * vy
    tile_r_1 = eu * wp.abs(vx) + ev * wp.abs(vy)
    if wp.abs(proj_dist_1) > r1 + tile_r_1:
        return False

    # Axis 2: OBB Minor Axis (-vy, vx)
    proj_dist_2 = du * -vy + dv * vx
    tile_r_2 = eu * wp.abs(-vy) + ev * wp.abs(vx)
    if wp.abs(proj_dist_2) > r2 + tile_r_2:
        return False

    return True


@dataclass
class TorchCamera:
    eye: torch.Tensor
    right: torch.Tensor
    up: torch.Tensor
    width: int
    height: int

    def to_warp(self):
        warp_cam = WarpCamera()
        warp_cam.eye = wp.vec3f(self.eye[0], self.eye[1], self.eye[2])
        warp_cam.right = wp.vec3f(self.right[0], self.right[1], self.right[2])
        warp_cam.up = wp.vec3f(self.up[0], self.up[1], self.up[2])
        warp_cam.width = self.width
        warp_cam.height = self.height
        return warp_cam

    def to_open3d(self) -> o3d.camera.PinholeCameraParameters:
        K = self.intrinsics_matrix()
        fx, fy, cx, cy = K[[0, 1, 0, 1], [0, 1, 2, 2]].tolist()

        params = o3d.camera.PinholeCameraParameters()
        params.intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=self.width, height=self.height, fx=fx, fy=fy, cx=cx, cy=cy
        )
        params.extrinsic = self.w2c().cpu().numpy()
        return params

    def to_device(self, device: torch.device | str):
        dev = torch.device(device)
        return TorchCamera(
            eye=self.eye.to(dev),
            right=self.right.to(dev),
            up=self.up.to(dev),
            width=self.width,
            height=self.height,
        )

    def get_ray_dir(self, i, j):
        forward = torch.cross(self.up, self.right, dim=-1)
        forward = forward / torch.norm(forward, dim=-1, keepdim=True)
        x = 2.0 * j / (float(self.width) - 1.0) - 1.0
        y = 1.0 - 2.0 * i / (float(self.height) - 1.0)
        return (
            x[:, None] * self.right[None, :]
            + y[:, None] * self.up[None, :]
            + forward[None, :]
        )

    def intrinsics_matrix(self) -> torch.Tensor:
        device = self.eye.device
        dtype = self.eye.dtype
        right_norm = torch.linalg.norm(self.right) + 1e-8
        up_norm = torch.linalg.norm(self.up) + 1e-8
        fx = 0.5 * (float(self.width) - 1.0) / right_norm
        fy = 0.5 * (float(self.height) - 1.0) / up_norm
        cx = 0.5 * (float(self.width) - 1.0)
        cy = 0.5 * (float(self.height) - 1.0)
        K = torch.tensor(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=dtype, device=device
        )
        return K

    def w2c(self) -> torch.Tensor:
        device = self.eye.device
        dtype = self.eye.dtype

        r = self.right / (torch.linalg.norm(self.right) + 1e-8)
        u = self.up / (torch.linalg.norm(self.up) + 1e-8)
        f = torch.cross(u, r, dim=-1)
        f = f / torch.linalg.norm(f + 1e-8)
        u = torch.cross(f, r, dim=-1)
        u = u / torch.linalg.norm(u + 1e-8)

        R = torch.stack([r, u, f], dim=0)
        t = -R @ self.eye

        M = torch.eye(4, dtype=dtype, device=device)
        M[:3, :3] = R
        M[:3, 3] = t
        return M

    def c2w(self) -> torch.Tensor:
        device = self.eye.device
        dtype = self.eye.dtype

        r = self.right / (torch.linalg.norm(self.right))
        u = self.up / (torch.linalg.norm(self.up))
        f = torch.cross(u, r)
        f = f / (torch.linalg.norm(f))
        u = torch.cross(f, r)
        u = u / (torch.linalg.norm(u))

        R_cw = torch.stack([r, u, f], dim=1)

        M = torch.eye(4, dtype=dtype, device=device)
        M[:3, :3] = R_cw
        M[:3, 3] = self.eye
        return M

    def projection_matrix(self) -> torch.Tensor:
        K = self.intrinsics_matrix()
        V = self.w2c()
        P = K @ V[:3, :]
        return P
