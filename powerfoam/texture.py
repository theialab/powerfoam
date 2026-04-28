import torch
import warp as wp

from .rendering_math import ray_plane_intersect, ray_plane_intersect_bwd

temp = wp.constant(10.0)
max_sites = wp.constant(8)


@wp.func
def plane_intersection_fwd(
    ray_origin: wp.vec3f,
    ray_direction: wp.vec3f,
    t_near: float,
    plane_origin: wp.vec3f,
    plane_normal: wp.vec3f,
    radius: float,
    hit: bool,
    sites: wp.array(dtype=wp.vec3f),
    values: wp.array(dtype=wp.vec4f),
    num_sites: int,
):
    shared_sites = wp.tile_load(sites, shape=(max_sites,))
    shared_values = wp.tile_load(values, shape=(max_sites,))

    if not hit:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, wp.vec3f(0.0, 0.0, 0.0), 0.0

    _t_surf, _dp = ray_plane_intersect(
        ray_origin, ray_direction, plane_origin, plane_normal
    )
    _t_query = t_near if _dp >= 0.0 else wp.max(t_near, _t_surf)
    _intersection_pt = ray_origin + _t_query * ray_direction

    height_sum = float(0.0)
    _weight_sum = float(0.0)
    for i in range(num_sites):
        site = shared_sites[i]
        dist_sq = wp.length_sq(_intersection_pt - site) / (radius * radius)
        weight = wp.exp(-temp * dist_sq)

        value = shared_values[i]
        height = value[3]
        height_sum += weight * height
        _weight_sum += weight

    _weight_sum = wp.max(_weight_sum, 1e-20)
    height_out = height_sum / _weight_sum

    t_surf, dp = ray_plane_intersect(
        ray_origin, ray_direction, plane_origin, plane_normal, height_out
    )
    t_query = t_near if dp >= 0.0 else wp.max(t_near, t_surf)
    intersection_pt = ray_origin + t_query * ray_direction

    rgb_sum = wp.vec3f(0.0, 0.0, 0.0)
    weight_sum = float(0.0)
    for i in range(num_sites):
        site = shared_sites[i]
        dist_sq = wp.length_sq(intersection_pt - site) / (radius * radius)
        weight = wp.exp(-temp * dist_sq)

        value = shared_values[i]
        rgb = wp.vec3f(value[0], value[1], value[2])
        rgb_sum += weight * rgb
        weight_sum += weight

    weight_sum = wp.max(weight_sum, 1e-20)
    rgb_out = rgb_sum / weight_sum

    return _t_surf, _dp, height_out, _weight_sum, t_surf, dp, rgb_out, weight_sum


@wp.func
def plane_intersection_bwd(
    ray_origin: wp.vec3f,
    ray_direction: wp.vec3f,
    _t_near: float,
    plane_origin: wp.vec3f,
    plane_normal: wp.vec3f,
    radius: float,
    hit: bool,
    sites: wp.array(dtype=wp.vec3f),
    values: wp.array(dtype=wp.vec4f),
    _dp: float,
    _t_surf: float,
    height_in: float,
    _weight_sum: float,
    dp: float,
    t_surf: float,
    rgb_in: wp.vec3f,
    weight_sum: float,
    dLdrgb_in: wp.vec3f,
    dLdt_surf_in: float,
    num_sites: int,
    sites_grad_out: wp.array(dtype=wp.vec3f),
    rgb_grad_out: wp.array(dtype=wp.vec3f),
    height_grad_out: wp.array(dtype=wp.float32),
):
    shared_sites = wp.tile_load(sites, shape=(max_sites,))
    shared_values = wp.tile_load(values, shape=(max_sites,))

    dLdplane_origin = wp.vec3f(0.0, 0.0, 0.0)
    dLdplane_normal = wp.vec3f(0.0, 0.0, 0.0)
    dLdheight_in = 0.0
    if not hit:
        return dLdplane_origin, dLdplane_normal, 0.0

    t_query = _t_near if dp >= 0.0 else wp.max(_t_near, t_surf)
    intersection_pt = ray_origin + t_query * ray_direction
    dLdintersection_pt = wp.vec3f(0.0, 0.0, 0.0)
    grad_dp_in = wp.dot(dLdrgb_in, rgb_in)

    for i in range(num_sites):
        site = shared_sites[i]
        dist_sq = wp.length_sq(intersection_pt - site) / (radius * radius)
        weight = wp.exp(-temp * dist_sq)
        if weight / weight_sum < 1e-3:
            continue
        dweightdsite = (2.0 * temp * weight * (intersection_pt - site)) / (
            radius * radius
        )

        rgb_grad = weight * dLdrgb_in / weight_sum
        rgb_grad_out[i] += rgb_grad

        value = shared_values[i]
        rgb = wp.vec3f(value[0], value[1], value[2])
        grad_dp = wp.dot(dLdrgb_in, rgb)
        site_grad = dweightdsite * (grad_dp - grad_dp_in) / weight_sum
        sites_grad_out[i] += site_grad

        dLdintersection_pt += -site_grad

    dLdt_query = wp.dot(dLdintersection_pt, ray_direction)
    dLd_t_near = dLdt_query if (dp >= 0.0 or _t_near >= t_surf) else 0.0
    dLdt_surf = dLdt_query if (dp < 0.0 and _t_near < t_surf) else 0.0
    dLdt_surf += dLdt_surf_in
    if wp.abs(dLdt_surf) > 0.0:
        dt_surf_dplane_origin, dt_surf_dplane_normal, dt_surf_dheight_in = (
            ray_plane_intersect_bwd(
                ray_origin, ray_direction, plane_origin, plane_normal, height_in
            )
        )
        dLdplane_origin += dt_surf_dplane_origin * dLdt_surf
        dLdplane_normal += dt_surf_dplane_normal * dLdt_surf
        dLdheight_in += dt_surf_dheight_in * dLdt_surf

    _t_query = _t_near if _dp >= 0.0 else wp.max(_t_near, _t_surf)
    _intersection_pt = ray_origin + _t_query * ray_direction
    dLd_intersection_pt = wp.vec3f(0.0, 0.0, 0.0)
    _grad_dp_in = dLdheight_in * height_in

    for i in range(num_sites):
        site = shared_sites[i]
        dist_sq = wp.length_sq(_intersection_pt - site) / (radius * radius)
        weight = wp.exp(-temp * dist_sq)
        if weight / weight_sum < 1e-3:
            continue
        dweightdsite = (2.0 * temp * weight * (_intersection_pt - site)) / (
            radius * radius
        )

        height_grad = weight * dLdheight_in / _weight_sum
        height_grad_out[i] += height_grad

        value = shared_values[i]
        height = value[3]
        _grad_dp = dLdheight_in * height
        site_grad = dweightdsite * (_grad_dp - _grad_dp_in) / _weight_sum
        sites_grad_out[i] += site_grad

        dLd_intersection_pt += -site_grad

    dLd_t_query = wp.dot(dLd_intersection_pt, ray_direction)
    dLd_t_near += dLd_t_query if (dp >= 0.0 or _t_near >= _t_surf) else 0.0
    dLd_t_surf = dLd_t_query if (dp < 0.0 and _t_near < _t_surf) else 0.0

    if wp.abs(dLd_t_surf) > 0.0:
        d_t_surfdplane_origin, d_t_surfdplane_normal, _ = ray_plane_intersect_bwd(
            ray_origin, ray_direction, plane_origin, plane_normal
        )
        dLdplane_origin += d_t_surfdplane_origin * dLd_t_surf
        dLdplane_normal += d_t_surfdplane_normal * dLd_t_surf

    return dLdplane_origin, dLdplane_normal, dLd_t_near


@wp.func
def soft_voronoi_fwd(
    coords: wp.vec3f,
    radius: float,
    sites: wp.array(dtype=wp.vec3f),
    values: wp.array(dtype=float),
    num_sites: int,
):
    value_out = float(0.0)
    weight_sum = float(0.0)
    for i in range(num_sites):
        site = sites[i]
        dist = wp.length(coords - site) / radius
        weight = wp.exp(-temp * dist**2.0)
        value = values[i]
        value_out += weight * value
        weight_sum += weight

    weight_sum = wp.max(weight_sum, 1e-20)
    value_out = value_out / weight_sum
    return value_out, weight_sum


@wp.func
def soft_voronoi_bwd(
    coords: wp.vec3f,
    radius: float,
    sites: wp.array(dtype=wp.vec3f),
    values: wp.array(dtype=float),
    value_out: float,
    weight_sum: float,
    dLdvalue_out: float,
    sites_grad: wp.array(dtype=wp.vec3f),
    values_grad: wp.array(dtype=float),
    num_sites: int,
):
    dp_out = dLdvalue_out * value_out  # dL/drgb . rgb
    dLdcoords = wp.vec3f(0.0, 0.0, 0.0)
    if weight_sum < 1e-20:
        return dLdcoords
    for i in range(num_sites):
        site = sites[i]
        dist = wp.length(coords - site) / radius
        weight = wp.exp(-temp * (dist**2.0))
        if weight / weight_sum < 1e-3:
            continue
        dweightdsite = (2.0 * temp * weight * (coords - site)) / (radius * radius)

        value = values[i]
        dp = dLdvalue_out * value
        values_grad[i] += weight * dLdvalue_out / weight_sum
        grad_site = dweightdsite * (dp - dp_out) / weight_sum
        sites_grad[i] += grad_site
        dLdcoords += -grad_site

    return dLdcoords


@wp.func
def soft_voronoi_fwd(
    coords: wp.vec3f,
    radius: float,
    sites: wp.array(dtype=wp.vec3f),
    values: wp.array(dtype=wp.vec3f),
    num_sites: int,
):
    value_out = wp.vec3f(0.0, 0.0, 0.0)
    weight_sum = float(0.0)
    for i in range(num_sites):
        site = sites[i]
        dist = wp.length(coords - site) / radius
        weight = wp.exp(-temp * dist**2.0)
        value = values[i]
        value_out += weight * value
        weight_sum += weight

    weight_sum = wp.max(weight_sum, 1e-20)
    value_out = value_out / weight_sum
    return value_out, weight_sum


@wp.func
def soft_voronoi_bwd(
    coords: wp.vec3f,
    radius: float,
    sites: wp.array(dtype=wp.vec3f),
    values: wp.array(dtype=wp.vec3f),
    value_out: wp.vec3f,
    weight_sum: float,
    dLdvalue_out: wp.vec3f,
    sites_grad: wp.array(dtype=wp.vec3f),
    values_grad: wp.array(dtype=wp.vec3f),
    num_sites: int,
):
    dp_out = wp.dot(dLdvalue_out, value_out)  # dL/drgb . rgb
    dLdcoords = wp.vec3f(0.0, 0.0, 0.0)
    if weight_sum < 1e-20:
        return dLdcoords
    for i in range(num_sites):
        site = sites[i]
        dist = wp.length(coords - site) / radius
        weight = wp.exp(-temp * dist**2.0)
        if weight / weight_sum < 1e-3:
            continue
        dweightdsite = (2.0 * temp * weight * (coords - site)) / (radius * radius)

        value = values[i]
        dp = wp.dot(dLdvalue_out, value)
        values_grad[i] += weight * dLdvalue_out / weight_sum
        grad_site = dweightdsite * (dp - dp_out) / weight_sum
        sites_grad[i] += grad_site
        dLdcoords += -grad_site

    return dLdcoords
