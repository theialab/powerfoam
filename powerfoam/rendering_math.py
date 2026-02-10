import warp as wp

_erf_cxx_code = """
    return erff(x);
    """


@wp.func_native(_erf_cxx_code)
def erf(x: float) -> float: ...


@wp.func
def dawson(x: float) -> float:
    p1 = 0.1349423927
    p2 = 0.0352304655
    p3 = 0.0138159073
    q1 = 0.8001569104
    q2 = 0.3190611301
    q3 = 0.0540828748

    y = x * x
    P = ((p3 * y + p2) * y + p1) * y + 1.0
    Q = (((2.0 * p3 * y + q3) * y + q2) * y + q1) * y + 1.0
    return x * P / Q


@wp.func
def erfi(x: float) -> float:
    return (2.0 / wp.sqrt(wp.pi)) * wp.exp(x * x) * dawson(x)


@wp.func
def ray_sphere_intersect(eye: wp.vec3f, dir: wp.vec3f, c: wp.vec3f, r: float):
    # Sphere equation: ||x - c||^2 = r^2
    oc = eye - c
    qb = 2.0 * wp.dot(oc, dir)
    qc = wp.dot(oc, oc) - r * r
    discriminant = qb * qb - 4.0 * qc

    if discriminant < 0:
        return False, 0.0, 0.0

    t_far = (-qb + wp.sqrt(discriminant)) / 2.0
    t_near = (-qb - wp.sqrt(discriminant)) / 2.0

    if t_near < 0.0 and t_far < 0.0:
        return False, 0.0, 0.0
    elif t_near < 0.0:
        return True, 0.0, t_far
    else:
        return True, t_near, t_far


@wp.func
def ray_sphere_intersect_bwd(eye: wp.vec3f, dir: wp.vec3f, c: wp.vec3f, r: float):
    # Sphere equation: ||x - c||^2 = r^2
    oc = eye - c
    qa = wp.dot(dir, dir)
    qb = 2.0 * wp.dot(oc, dir)
    qc = wp.dot(oc, oc) - r * r
    discriminant = qb * qb - 4.0 * qa * qc

    dt_neardc = wp.vec3f(0.0)
    dt_fardc = wp.vec3f(0.0)
    dt_neardr = 0.0
    dt_fardr = 0.0

    if discriminant < 0:
        return dt_neardc, dt_fardc, dt_neardr, dt_fardr

    t_far = (-qb + wp.sqrt(discriminant)) / (2.0 * qa)
    t_near = (-qb - wp.sqrt(discriminant)) / (2.0 * qa)

    if t_near < 0.0 and t_far < 0.0:
        return dt_neardc, dt_fardc, dt_neardr, dt_fardr
    elif t_near < 0.0:
        # only t_far is valid
        dt_fardc = dir / qa + (2.0 * qa * oc - qb * dir) / (qa * wp.sqrt(discriminant))
        dt_fardr = 2.0 * r / wp.sqrt(discriminant)
    else:
        # both t_near and t_far are valid
        dt_neardc = dir / qa - (2.0 * qa * oc - qb * dir) / (qa * wp.sqrt(discriminant))
        dt_neardr = -2.0 * r / wp.sqrt(discriminant)
        dt_fardc = dir / qa + (2.0 * qa * oc - qb * dir) / (qa * wp.sqrt(discriminant))
        dt_fardr = 2.0 * r / wp.sqrt(discriminant)

    return dt_neardc, dt_fardc, dt_neardr, dt_fardr


@wp.func
def ray_pface_intersect(
    eye: wp.vec3f,
    dir: wp.vec3f,
    center: wp.vec3f,
    radius: float,
    adj_center: wp.vec3f,
    adj_radius: float,
):
    # Power face equation: face_n . x = face_offset
    face_n = adj_center - center
    face_offset = 0.5 * (
        wp.dot(adj_center, adj_center)
        - wp.dot(center, center)
        + radius * radius
        - adj_radius * adj_radius
    )

    dp = wp.dot(dir, face_n)
    t = (face_offset - wp.dot(eye, face_n)) / dp
    return t, dp


@wp.func
def ray_pface_intersect_diff(
    eye: wp.vec3f,
    dir: wp.vec3f,
    diff: wp.vec3f,
    pm_diff: float,
):
    # Power face equation with precomputed difference
    # face_n = diff
    # face_offset = pm_diff

    dp = wp.dot(dir, diff)
    t = (pm_diff - wp.dot(eye, diff)) / dp
    return t, dp


@wp.func
def ray_pface_intersect_bwd(
    eye: wp.vec3f,
    dir: wp.vec3f,
    center: wp.vec3f,
    radius: float,
    adj_center: wp.vec3f,
    adj_radius: float,
):
    # Power face equation: face_n . x = face_offset
    face_n = adj_center - center
    face_offset = 0.5 * (
        wp.dot(adj_center, adj_center)
        - wp.dot(center, center)
        + radius * radius
        - adj_radius * adj_radius
    )

    num = face_offset - wp.dot(eye, face_n)
    dp = wp.dot(dir, face_n)
    dtdcenter = (-center + eye) / dp + num * dir / dp / dp
    dtdadjcenter = (adj_center - eye) / dp - num * dir / dp / dp
    dtdradius = radius / dp
    dtdadjradius = -adj_radius / dp
    return dtdcenter, dtdradius, dtdadjcenter, dtdadjradius


@wp.func
def ray_plane_intersect(
    eye: wp.vec3f, dir: wp.vec3f, p: wp.vec3f, n: wp.vec3f, h: wp.float32 = 0.0
):
    # Plane equation: n . x = n . p + h
    dp = wp.dot(n, dir)
    t = (wp.dot(p - eye, n) + h) / dp
    return t, dp


@wp.func
def ray_plane_intersect_bwd(
    eye: wp.vec3f, dir: wp.vec3f, p: wp.vec3f, n: wp.vec3f, h: wp.float32 = 0.0
):
    # Plane equation: n . x = n . p + h
    dp = wp.dot(n, dir)
    dtdp = n / (dp + 1e-6)
    dtdh = 1.0 / (dp + 1e-6)
    dtdn = (p - eye) / (dp + 1e-6) - (wp.dot(p - eye, n) + h) * dir / (dp * dp + 1e-6)
    return dtdp, dtdn, dtdh


@wp.func
def density_integral(
    eye: wp.vec3f,
    dir: wp.vec3f,
    t_near: float,
    t_far: float,
    center: wp.vec3f,
    radius: float,
    sigma: float,
):
    m = (t_far - t_near) * sigma
    return -m


@wp.func
def density_integral_bwd(
    eye: wp.vec3f,
    dir: wp.vec3f,
    t_near: float,
    t_far: float,
    center: wp.vec3f,
    radius: float,
    sigma: float,
):
    dmdt_near = -sigma
    dmdt_far = sigma
    dmdcenter = wp.vec3f(0.0)
    dmdradius = 0.0
    dmdsigma = t_far - t_near
    dmdh = 0.0
    return (
        -dmdt_near,
        -dmdt_far,
        -dmdcenter,
        -dmdradius,
        -dmdsigma,
    )


# @wp.func
# def density_integral(
#     eye: wp.vec3f,
#     dir: wp.vec3f,
#     t_near: float,
#     t_far: float,
#     center: wp.vec3f,
#     radius: float,
#     sigma: float,
#     h: float,
# ):
#     c_ray = center - eye
#     t_center = wp.dot(c_ray, dir)
#     c_dist = wp.length(c_ray - t_center * dir)

#     u0 = (t_near - t_center) / radius
#     u1 = (t_far - t_center) / radius
#     d = c_dist / radius
#     d2 = d * d

#     if wp.abs(h) < 1e-3:
#         h = 1e-3 * wp.sign(h)
#     if wp.abs(h) > 1e2:
#         h = 1e2 * wp.sign(h)

#     hrt = wp.sqrt(wp.abs(h))

#     if h < 0:
#         a0 = erf(hrt * u0) - erf(hrt * u1)
#     else:
#         a0 = erfi(hrt * u0) - erfi(hrt * u1)

#     eh = wp.exp(h)
#     a1 = 1.0 / hrt
#     a2 = 1.0 / (eh - 1.0)
#     a3 = wp.exp(d2 * h)
#     a4 = a0 * a1 * a2 * a3 * wp.sqrt(wp.pi) / 2.0
#     a5 = a2 * eh * (u1 - u0)
#     F = a4 + a5

#     m = sigma * radius * F

#     return -wp.max(m, 0.0)


# @wp.func
# def density_integral_bwd(
#     eye: wp.vec3f,
#     dir: wp.vec3f,
#     t_near: float,
#     t_far: float,
#     center: wp.vec3f,
#     radius: float,
#     sigma: float,
#     h: float,
# ):
#     c_ray = center - eye
#     t_center = wp.dot(c_ray, dir)
#     c_dist = wp.length(c_ray - t_center * dir)

#     u0 = (t_near - t_center) / radius
#     u1 = (t_far - t_center) / radius
#     d = c_dist / radius
#     d2 = d * d

#     if wp.abs(h) < 1e-3:
#         h = 1e-3 * wp.sign(h)
#     if wp.abs(h) > 1e2:
#         h = 1e2 * wp.sign(h)

#     hrt = wp.sqrt(wp.abs(h))

#     if h < 0:
#         a0 = erf(hrt * u0) - erf(hrt * u1)
#     else:
#         a0 = erfi(hrt * u0) - erfi(hrt * u1)

#     eh = wp.exp(h)
#     a1 = 1.0 / hrt
#     a2 = 1.0 / (eh - 1.0)
#     a3 = wp.exp(d2 * h)
#     a4 = a0 * a1 * a2 * a3 * wp.sqrt(wp.pi) / 2.0
#     a5 = a2 * eh * (u1 - u0)
#     F = a4 + a5

#     dmdsigma = radius * F
#     dmdradius = sigma * F
#     dmdF = sigma * radius

#     dmda5 = dmdF
#     dmda2 = dmda5 * (u1 - u0) * eh
#     dmdeh = dmda5 * a2 * (u1 - u0)
#     dmdu1 = dmda5 * a2 * eh
#     dmdu0 = -dmda5 * a2 * eh

#     dmda4 = dmdF
#     dmda3 = dmda4 * a0 * a1 * a2 * wp.sqrt(wp.pi) / 2.0
#     dmda2 += dmda4 * a0 * a1 * a3 * wp.sqrt(wp.pi) / 2.0
#     dmda1 = dmda4 * a0 * a2 * a3 * wp.sqrt(wp.pi) / 2.0
#     dmda0 = dmda4 * a1 * a2 * a3 * wp.sqrt(wp.pi) / 2.0

#     dmd2 = dmda3 * h * wp.exp(d2 * h)
#     dmdh = dmda3 * d2 * wp.exp(d2 * h)

#     dmdeh += dmda2 * (-1.0) / ((eh - 1.0) * (eh - 1.0))

#     dmdhrt = dmda1 * (-1.0) / (hrt * hrt)

#     dmdh += dmdeh * wp.exp(h)

#     if h < 0:
#         derf0 = 2.0 * wp.exp(-(hrt * u0) * (hrt * u0)) / wp.sqrt(wp.pi)
#         derf1 = -2.0 * wp.exp(-(hrt * u1) * (hrt * u1)) / wp.sqrt(wp.pi)

#         dmdu0 += dmda0 * derf0 * hrt
#         dmdu1 += dmda0 * derf1 * hrt
#         dmdhrt += dmda0 * (derf0 * u0 + derf1 * u1)
#     else:
#         derfi0 = 2.0 * wp.exp((hrt * u0) * (hrt * u0)) / wp.sqrt(wp.pi)
#         derfi1 = -2.0 * wp.exp((hrt * u1) * (hrt * u1)) / wp.sqrt(wp.pi)

#         dmdu0 += dmda0 * derfi0 * hrt
#         dmdu1 += dmda0 * derfi1 * hrt
#         dmdhrt += dmda0 * (derfi0 * u0 + derfi1 * u1)

#     dmdh += dmdhrt * wp.sign(h) / (2.0 * hrt)

#     dmdd = dmd2 * 2.0 * d

#     dmdc_dist = dmdd / radius
#     dmdt_near = dmdu0 / radius
#     dmdt_far = dmdu1 / radius
#     dmdt_center = -(dmdu0 + dmdu1) / radius
#     dmdt_center += dmdc_dist * -wp.dot(c_ray - t_center * dir, dir) / (c_dist + 1e-6)
#     dmdc_ray = dmdc_dist * (c_ray - t_center * dir) / (c_dist + 1e-6)
#     dmdc_ray += dmdt_center * dir
#     dmdcenter = dmdc_ray

#     return (
#         -dmdt_near,
#         -dmdt_far,
#         -dmdcenter,
#         -dmdradius,
#         -dmdsigma,
#         -dmdh,
#     )
