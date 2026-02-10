import torch
import warp as wp


class SphericalVoronoiGradFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, spherical_voronoi, points, camera, att_sites, att_values, att_temps
    ):
        return spherical_voronoi._forward(
            ctx, points, camera, att_sites, att_values, att_temps
        )

    @staticmethod
    def backward(ctx, *args):
        return ctx.spherical_voronoi._backward(ctx, *args)


class SphericalVoronoi:
    def __init__(self, args, device, camera, attr_dtype="float"):
        self.device = device
        self.args = args
        sv_dof = args.sv_dof
        self.sv_dof = sv_dof
        if attr_dtype == "float":
            scalar = wp.float32
            self.tscalar = torch.float32
            vec3s = wp.vec3f
        elif attr_dtype == "half":
            scalar = wp.float16
            self.tscalar = torch.float16
            vec3s = wp.vec3h
        else:
            raise ValueError(f"Unsupported attribute dtype: {attr_dtype}")

        K = camera.intrinsics_matrix()
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]

        x_max = max(float(cx), float(camera.width) - float(cx)) / float(fx)
        y_max = max(float(cy), float(camera.height) - float(cy)) / float(fy)
        tan_theta = float((x_max**2 + y_max**2) ** 0.5 * 1.1)
        fov_cos_cutoff = 1.0 / ((1.0 + tan_theta**2) ** 0.5)

        @wp.kernel(enable_backward=False)
        def spherical_voronoi_fwd_kernel(
            points: wp.array(dtype=wp.vec3f),
            camera_origin: wp.vec3f,
            camera_forward: wp.vec3f,
            att_sites: wp.array2d(dtype=vec3s),
            att_values: wp.array2d(dtype=vec3s),
            att_temps: wp.array2d(dtype=scalar),
            num_points: int,
            value_out: wp.array(dtype=vec3s),
            weights_sum_out: wp.array(dtype=scalar),
        ):
            thread_idx = wp.tid()
            if thread_idx >= num_points:
                return

            point = points[thread_idx]
            direction = wp.normalize(point - camera_origin)
            if wp.dot(direction, camera_forward) < fov_cos_cutoff:
                value_out[thread_idx] = vec3s(scalar(0.5), scalar(0.5), scalar(0.5))
                weights_sum_out[thread_idx] = scalar(0.0)
                return

            weights_sum = float(0.0)
            value_sum = wp.vec3f(0.0)
            for i in range(sv_dof):
                _axis = att_sites[i, thread_idx]
                _val = att_values[i, thread_idx]
                _temp = att_temps[i, thread_idx]

                axis = wp.vec3f(float(_axis[0]), float(_axis[1]), float(_axis[2]))
                val = wp.vec3f(float(_val[0]), float(_val[1]), float(_val[2]))
                temp = float(_temp)

                dist = wp.length(direction - axis)
                weight = wp.exp(-temp * dist)
                weights_sum += weight
                value_sum += weight * val

            weights_sum_out[thread_idx] = scalar(weights_sum)
            value = value_sum / weights_sum + wp.vec3f(0.5, 0.5, 0.5)
            value = wp.vec3f(
                wp.max(value[0], 0.0), wp.max(value[1], 0.0), wp.max(value[2], 0.0)
            )
            value_out[thread_idx] = vec3s(
                scalar(value[0]), scalar(value[1]), scalar(value[2])
            )

        self.spherical_voronoi_fwd_kernel = spherical_voronoi_fwd_kernel

        @wp.kernel(enable_backward=False)
        def spherical_voronoi_bwd_kernel(
            points: wp.array(dtype=wp.vec3f),
            camera_origin: wp.vec3f,
            camera_forward: wp.vec3f,
            att_sites: wp.array2d(dtype=vec3s),
            att_values: wp.array2d(dtype=vec3s),
            att_temps: wp.array2d(dtype=scalar),
            weights_sum_in: wp.array(dtype=scalar),
            value_in: wp.array(dtype=vec3s),
            grad_value_in: wp.array(dtype=vec3s),
            num_points: int,
            grad_points_out: wp.array(dtype=wp.vec3f),
            grad_att_sites_out: wp.array2d(dtype=vec3s),
            grad_att_values_out: wp.array2d(dtype=vec3s),
            grad_att_temps_out: wp.array2d(dtype=scalar),
        ):
            thread_idx = wp.tid()
            if thread_idx >= num_points:
                return

            point = points[thread_idx]
            direction = wp.normalize(point - camera_origin)
            ddirectiondpoint = (
                1.0
                / wp.length(point - camera_origin)
                * (wp.identity(3, dtype=wp.float32) - wp.outer(direction, direction))
            )
            if wp.dot(direction, camera_forward) < fov_cos_cutoff:
                return

            _weights_sum = weights_sum_in[thread_idx]
            _value = value_in[thread_idx]
            _dLdvalue = grad_value_in[thread_idx]
            weights_sum = float(_weights_sum)
            value = wp.vec3f(float(_value[0]), float(_value[1]), float(_value[2]))
            dLdvalue = wp.vec3f(
                float(_dLdvalue[0]), float(_dLdvalue[1]), float(_dLdvalue[2])
            )
            for i in range(3):
                if value[i] <= 0.0:
                    dLdvalue[i] = 0.0
            value = value - wp.vec3f(0.5, 0.5, 0.5)
            dp_out = wp.dot(dLdvalue, value)

            # grad_direction = wp.vec3f(0.0)
            for i in range(sv_dof):
                _axis = att_sites[i, thread_idx]
                _val = att_values[i, thread_idx]
                _temp = att_temps[i, thread_idx]

                axis = wp.vec3f(float(_axis[0]), float(_axis[1]), float(_axis[2]))
                val = wp.vec3f(float(_val[0]), float(_val[1]), float(_val[2]))
                temp = float(_temp)

                dist = wp.length(direction - axis)
                weight = wp.exp(-temp * dist)
                if weight / weights_sum < 1e-3:
                    continue

                dLdval = weight * dLdvalue / weights_sum
                grad_att_values_out[i, thread_idx] = vec3s(
                    scalar(dLdval[0]), scalar(dLdval[1]), scalar(dLdval[2])
                )

                dweightdsite = temp * weight * (direction - axis) / dist
                dweightdtemp = -dist * weight
                dp = wp.dot(dLdvalue, val)
                dLdaxis = dweightdsite * (dp - dp_out) / weights_sum
                dLdtemp = dweightdtemp * (dp - dp_out) / weights_sum
                grad_att_sites_out[i, thread_idx] = vec3s(
                    scalar(dLdaxis[0]), scalar(dLdaxis[1]), scalar(dLdaxis[2])
                )
                grad_att_temps_out[i, thread_idx] = scalar(dLdtemp)

                # grad_direction += -dweightdsite * (dp - dp_out) / weights_sum

            # grad_point = wp.dot(ddirectiondpoint, grad_direction)
            # grad_points_out[thread_idx] = grad_point

        self.spherical_voronoi_bwd_kernel = spherical_voronoi_bwd_kernel

    def _forward(self, grad_ctx, points, camera, att_sites, att_values, att_temps):
        with wp.ScopedDevice(str(self.device)):
            torch_stream = torch.cuda.current_stream()
            wp_stream = wp.stream_from_torch(torch_stream)
            wp.set_stream(wp_stream)

            num_points = points.shape[0]

            values_out = torch.empty(
                points.shape[0], 3, device=self.device, dtype=self.tscalar
            )
            weights_sum_out = torch.empty(
                points.shape[0], device=self.device, dtype=self.tscalar
            )

            camera_origin = camera.eye
            camera_forward = torch.cross(camera.up, camera.right, dim=-1)
            camera_forward = camera_forward / torch.norm(
                camera_forward, dim=-1, keepdim=True
            )

            wp.launch(
                kernel=self.spherical_voronoi_fwd_kernel,
                dim=points.shape[0],
                inputs=[
                    points.detach(),
                    camera_origin,
                    camera_forward,
                    att_sites.detach(),
                    att_values.detach(),
                    att_temps.detach(),
                    num_points,
                    values_out,
                    weights_sum_out,
                ],
                block_dim=256,
            )

            grad_ctx.spherical_voronoi = self
            grad_ctx.save_for_backward(
                points,
                camera_origin,
                camera_forward,
                att_sites,
                att_values,
                att_temps,
                values_out,
                weights_sum_out,
            )

            return values_out

    def forward(self, points, camera, att_sites, att_values, att_temps):
        return SphericalVoronoiGradFn.apply(
            self, points, camera, att_sites, att_values, att_temps
        )

    def _backward(self, grad_ctx, grad_values):
        with wp.ScopedDevice(str(self.device)):
            torch_stream = torch.cuda.current_stream()
            wp_stream = wp.stream_from_torch(torch_stream)
            wp.set_stream(wp_stream)

            (
                points,
                camera_origin,
                camera_forward,
                att_sites,
                att_values,
                att_temps,
                values_in,
                weights_sum_in,
            ) = grad_ctx.saved_tensors

            num_points = points.shape[0]

            grad_points_out = torch.zeros_like(points)
            grad_att_sites_out = torch.zeros_like(att_sites)
            grad_att_values_out = torch.zeros_like(att_values)
            grad_att_temps_out = torch.zeros_like(att_temps)

            wp.launch(
                kernel=self.spherical_voronoi_bwd_kernel,
                dim=points.shape[0],
                inputs=[
                    points.detach(),
                    camera_origin,
                    camera_forward,
                    att_sites.detach(),
                    att_values.detach(),
                    att_temps.detach(),
                    weights_sum_in.detach(),
                    values_in.detach(),
                    grad_values.detach(),
                    num_points,
                    grad_points_out,
                    grad_att_sites_out,
                    grad_att_values_out,
                    grad_att_temps_out,
                ],
                block_dim=256,
            )

            grad_att_sv = (
                grad_att_sites_out,
                grad_att_values_out,
                grad_att_temps_out,
            )

            return (
                None,  # spherical_voronoi
                None,  # points
                None,  # camera
                grad_att_sites_out,
                grad_att_values_out,
                grad_att_temps_out,
            )
