import torch
import warp as wp


@wp.func
def grid_interpolate_fwd(
    coords: wp.vec3f,
    center: wp.vec3f,
    radius: float,
    normal: wp.vec3f,
    tangent: wp.vec3f,
    grid_rgb: wp.array(dtype=wp.vec3f),
    grid_resolution: int,
) -> wp.vec3f:
    bitangent = wp.cross(normal, tangent)

    offset = coords - center
    u_ = wp.dot(offset, tangent) / radius
    u = (u_ + 1.0) * 0.5 * float(grid_resolution - 1)
    v_ = wp.dot(offset, bitangent) / radius
    v = (v_ + 1.0) * 0.5 * float(grid_resolution - 1)

    u0 = wp.clamp(int(wp.floor(u)), 0, grid_resolution - 2)
    v0 = wp.clamp(int(wp.floor(v)), 0, grid_resolution - 2)
    u1, v1 = u0 + 1, v0 + 1

    weights = wp.vec4f()
    weights[0] = (float(u1) - u) * (float(v1) - v)
    weights[1] = (u - float(u0)) * (float(v1) - v)
    weights[2] = (float(u1) - u) * (v - float(v0))
    weights[3] = (u - float(u0)) * (v - float(v0))

    rgb = wp.vec3f(0.0, 0.0, 0.0)
    rgb += weights[0] * grid_rgb[u0 * grid_resolution + v0]
    rgb += weights[1] * grid_rgb[u1 * grid_resolution + v0]
    rgb += weights[2] * grid_rgb[u0 * grid_resolution + v1]
    rgb += weights[3] * grid_rgb[u1 * grid_resolution + v1]

    return rgb


@wp.func
def grid_interpolate_bwd(
    coords: wp.vec3f,
    center: wp.vec3f,
    radius: float,
    normal: wp.vec3f,
    tangent: wp.vec3f,
    dLdrgb: wp.vec3f,
    grid_rgb: wp.array(dtype=wp.vec3f),
    grid_rgb_grad: wp.array(dtype=wp.vec3f),
    grid_resolution: int,
):
    bitangent = wp.cross(normal, tangent)

    offset = coords - center
    u_ = wp.dot(offset, tangent) / radius
    u = (u_ + 1.0) * 0.5 * float(grid_resolution - 1)
    v_ = wp.dot(offset, bitangent) / radius
    v = (v_ + 1.0) * 0.5 * float(grid_resolution - 1)

    u0 = wp.clamp(int(wp.floor(u)), 0, grid_resolution - 2)
    v0 = wp.clamp(int(wp.floor(v)), 0, grid_resolution - 2)
    u1, v1 = u0 + 1, v0 + 1

    weights = wp.vec4f()
    weights[0] = (float(u1) - u) * (float(v1) - v)
    weights[1] = (u - float(u0)) * (float(v1) - v)
    weights[2] = (float(u1) - u) * (v - float(v0))
    weights[3] = (u - float(u0)) * (v - float(v0))

    grid_rgb_grad[u0 * grid_resolution + v0] += weights[0] * dLdrgb
    grid_rgb_grad[u1 * grid_resolution + v0] += weights[1] * dLdrgb
    grid_rgb_grad[u0 * grid_resolution + v1] += weights[2] * dLdrgb
    grid_rgb_grad[u1 * grid_resolution + v1] += weights[3] * dLdrgb

    dp_1 = wp.dot(grid_rgb[u0 * grid_resolution + v0], dLdrgb)
    dp_2 = wp.dot(grid_rgb[u1 * grid_resolution + v0], dLdrgb)
    dp_3 = wp.dot(grid_rgb[u0 * grid_resolution + v1], dLdrgb)
    dp_4 = wp.dot(grid_rgb[u1 * grid_resolution + v1], dLdrgb)

    val_1 = (
        dp_1 * (v - float(v1))
        + dp_2 * (float(v1) - v)
        + dp_3 * (float(v0) - v)
        + dp_4 * (v - float(v0))
    )
    val_2 = (
        dp_1 * (u - float(u1))
        + dp_2 * (float(u0) - u)
        + dp_3 * (float(u1) - u)
        + dp_4 * (u - float(u0))
    )

    dLdu = val_1 * 0.5 * float(grid_resolution - 1)
    dLdv = val_2 * 0.5 * float(grid_resolution - 1)

    dLdcoords = dLdu * (tangent / radius) + dLdv * (bitangent / radius)
    dLdcenter = -dLdcoords
    dLdradius = -(dLdu * u_ + dLdv * v_) / radius

    return dLdcoords, dLdcenter, dLdradius
