import ctypes
import threading
import time
import math
import numpy as np
import torch
from OpenGL import GL
import glfw
from imgui_bundle import imgui, hello_imgui
import warp as wp
from .camera import TorchCamera
from .rasterize import VisOptions


VIS_MODES = ["RGB", "Depth", "Normal", "Alpha", "Intersections"]
COLOR_MAPS = ["Gray", "Viridis", "Inferno", "Turbo"]
RENDER_MODES = ["Rasterize"]
RENDER_MODE_KEYS = ["rasterize"]


# ── Warp colormap functions ──────────────────────────────────────────────────


@wp.func
def turbo_colormap_wp(t: float) -> wp.vec3f:
    t = wp.clamp(t, 0.0, 1.0)
    r = wp.clamp(
        0.13572138
        + t
        * (
            4.6153926
            + t
            * (
                -42.66032258
                + t * (130.70100102 + t * (-152.94239396 + t * 59.28637943))
            )
        ),
        0.0,
        1.0,
    )
    g = wp.clamp(
        0.09140261
        + t
        * (
            2.19418839
            + t * (4.84296658 + t * (-14.18503333 + t * (4.27729857 + t * 2.82956604)))
        ),
        0.0,
        1.0,
    )
    b = wp.clamp(
        0.10667330
        + t
        * (
            12.64194608
            + t
            * (-60.58204836 + t * (110.36276771 + t * (-89.90310912 + t * 27.34824973)))
        ),
        0.0,
        1.0,
    )
    return wp.vec3f(r, g, b)


@wp.func
def viridis_colormap_wp(t: float) -> wp.vec3f:
    t = wp.clamp(t, 0.0, 1.0)
    c0 = wp.vec3f(0.2777273272234177, 0.005407344544966578, 0.3340998053353061)
    c1 = wp.vec3f(0.1050930431085774, 1.404613529898575, 1.749760295930354)
    c2 = wp.vec3f(-0.3308618287255563, 0.214847559468213, 0.09509516302823659)
    c3 = wp.vec3f(-4.634230498983486, -5.799100973351585, -19.33244095627987)
    c4 = wp.vec3f(6.228269936347081, 14.17993336680509, 56.69055260068105)
    c5 = wp.vec3f(4.776384997670612, -13.74514537774601, -65.35303263337234)
    c6 = wp.vec3f(-5.435455855934631, 4.645852612178535, 26.3124352495832)
    color = c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))))
    r = wp.clamp(color[0], 0.0, 1.0)
    g = wp.clamp(color[1], 0.0, 1.0)
    b = wp.clamp(color[2], 0.0, 1.0)
    return wp.vec3f(r, g, b)


@wp.func
def inferno_colormap_wp(t: float) -> wp.vec3f:
    t = wp.clamp(t, 0.0, 1.0)
    c0 = wp.vec3f(0.0002189403691192265, 0.001651004631001012, -0.01948089843709184)
    c1 = wp.vec3f(0.1065134194856116, 0.5639564367884091, 3.932712388889277)
    c2 = wp.vec3f(11.60249308247187, -3.972853965665698, -15.9423941062914)
    c3 = wp.vec3f(-41.70399613139459, 17.43639888205313, 44.35414519872813)
    c4 = wp.vec3f(77.162935699427, -33.40235894210092, -81.80730925738993)
    c5 = wp.vec3f(-73.76882330882026, 32.62606426397723, 73.20951985803202)
    c6 = wp.vec3f(27.16326261170523, -12.24266895238567, -23.07032500287172)
    color = c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))))
    r = wp.clamp(color[0], 0.0, 1.0)
    g = wp.clamp(color[1], 0.0, 1.0)
    b = wp.clamp(color[2], 0.0, 1.0)
    return wp.vec3f(r, g, b)


@wp.func
def apply_colormap_wp(t: float, color_map: int) -> wp.vec3f:
    if color_map == 1:
        return viridis_colormap_wp(t)
    elif color_map == 2:
        return inferno_colormap_wp(t)
    elif color_map == 3:
        return turbo_colormap_wp(t)
    # 0 = Gray (default)
    return wp.vec3f(t, t, t)


# ── Compose options struct ───────────────────────────────────────────────────


@wp.struct
class ComposeOptions:
    vis_mode: wp.int32
    color_map: wp.int32
    effective_max_depth: float
    max_intersections: wp.int32
    checker_bg: wp.int32
    bg_color: wp.vec3f
    width: wp.int32
    height: wp.int32


# ── Compose kernel ───────────────────────────────────────────────────────────


@wp.kernel
def compose_kernel(
    color: wp.array2d(dtype=wp.vec3f),
    depth: wp.array2d(dtype=wp.float32),
    normal: wp.array2d(dtype=wp.vec3f),
    alpha: wp.array2d(dtype=wp.float32),
    intersections: wp.array2d(dtype=wp.int32),
    options: ComposeOptions,
    output: wp.array(dtype=wp.uint8),
):
    i, j = wp.tid()
    if i >= options.height or j >= options.width:
        return

    rgb = wp.vec3f(0.0, 0.0, 0.0)

    if options.vis_mode == 0:
        # RGB
        c = color[i, j]
        if options.checker_bg != 0:
            check = (i / 16 + j / 16) % 2
            bg_val = float(check) * 0.5 + 0.25
            bg = wp.vec3f(bg_val, bg_val, bg_val)
            a = wp.clamp(alpha[i, j], 0.0, 1.0)
            rgb = c + bg * (1.0 - a)
        else:
            rgb = c
    elif options.vis_mode == 1:
        # Depth
        d = wp.clamp(
            depth[i, j] / wp.max(options.effective_max_depth, 1.0e-6), 0.0, 1.0
        )
        rgb = apply_colormap_wp(d, options.color_map)
    elif options.vis_mode == 2:
        # Normal
        n = normal[i, j]
        n_len = wp.max(wp.length(n), 1.0e-6)
        n = n / n_len
        rgb = n * 0.5 + wp.vec3f(0.5, 0.5, 0.5)
    elif options.vis_mode == 3:
        # Alpha
        a = wp.clamp(alpha[i, j], 0.0, 1.0)
        rgb = wp.vec3f(a, a, a)
    elif options.vis_mode == 4:
        # Intersections
        max_int = float(wp.max(options.max_intersections, 1))
        i_norm = wp.clamp(float(intersections[i, j]) / max_int, 0.0, 1.0)
        rgb = apply_colormap_wp(i_norm, options.color_map)

    r = wp.uint8(wp.clamp(rgb[0] * 255.0, 0.0, 255.0))
    g = wp.uint8(wp.clamp(rgb[1] * 255.0, 0.0, 255.0))
    b = wp.uint8(wp.clamp(rgb[2] * 255.0, 0.0, 255.0))

    offset = (i * options.width + j) * 4
    output[offset + 0] = r
    output[offset + 1] = g
    output[offset + 2] = b
    output[offset + 3] = wp.uint8(255)


# ── Helpers ──────────────────────────────────────────────────────────────────


def rotate_vec(v, axis, angle):
    axis = axis / np.linalg.norm(axis)
    c = math.cos(angle)
    s = math.sin(angle)
    return v * c + np.cross(axis, v) * s + axis * np.dot(axis, v) * (1 - c)


# ── Camera controller ───────────────────────────────────────────────────────


class CameraController:
    def __init__(
        self, camera: TorchCamera, world_up=None, move_speed=0.1, rotate_speed=0.003
    ):
        self.eye = camera.eye.detach().cpu().numpy()
        self.right = camera.right.detach().cpu().numpy()
        self.up = camera.up.detach().cpu().numpy()
        self.move_speed = move_speed
        self.rotate_speed = rotate_speed
        self.width = camera.width
        self.height = camera.height
        self._up_mag = np.linalg.norm(self.up)

        # Global up direction for horizon-locked rotation
        if world_up is not None:
            self.world_up = world_up / np.linalg.norm(world_up)
        else:
            self.world_up = self.up / np.linalg.norm(self.up)

        # Derive initial forward and re-orthogonalize to world_up
        forward = np.cross(self.up, self.right)
        forward /= np.linalg.norm(forward)
        self._reorthogonalize(forward)

    def _reorthogonalize(self, forward):
        """Re-derive right and up from forward + world_up to keep horizon level."""
        forward = forward / np.linalg.norm(forward)
        self.right = np.cross(forward, self.world_up)
        r_len = np.linalg.norm(self.right)
        if r_len < 1e-6:
            # Looking straight up/down -- pick an arbitrary right
            self.right = np.array([1.0, 0.0, 0.0])
        else:
            self.right = self.right / r_len
        self.up = np.cross(self.right, forward)
        self.up /= np.linalg.norm(self.up)

    def update(self, io: imgui.IO):
        if not io.want_capture_mouse:
            if io.mouse_down[0]:
                dx = io.mouse_delta.x
                dy = io.mouse_delta.y

                forward = np.cross(self.up, self.right)
                forward /= np.linalg.norm(forward)

                # Yaw around world up
                angle_x = -dx * self.rotate_speed
                forward = rotate_vec(forward, self.world_up, angle_x)

                # Pitch around right
                angle_y = -dy * self.rotate_speed
                forward = rotate_vec(forward, self.right, angle_y)

                # Re-derive right/up from forward + world_up
                self._reorthogonalize(forward)

        # Movement
        forward = np.cross(self.up, self.right)
        forward /= np.linalg.norm(forward)

        move = np.zeros(3)
        if imgui.is_key_down(imgui.Key.w):
            move += forward
        if imgui.is_key_down(imgui.Key.s):
            move -= forward
        if imgui.is_key_down(imgui.Key.d):
            move += self.right
        if imgui.is_key_down(imgui.Key.a):
            move -= self.right
        cam_up = self.up / max(np.linalg.norm(self.up), 1e-8)
        if imgui.is_key_down(imgui.Key.left_shift):
            move += cam_up
        if imgui.is_key_down(imgui.Key.left_ctrl):
            move -= cam_up

        if np.linalg.norm(move) > 0:
            move = move / np.linalg.norm(move)
            self.eye += move * self.move_speed * io.delta_time * 60.0

    def get_eval_camera(self, device):
        # Reconstruct right/up with correct magnitudes (keep vfov fixed, adjust hfov for aspect)
        up_dir = self.up / np.linalg.norm(self.up)
        right_dir = self.right / np.linalg.norm(self.right)
        aspect = self.width / max(self.height, 1)
        up_scaled = up_dir * self._up_mag
        right_scaled = right_dir * self._up_mag * aspect
        return TorchCamera(
            eye=torch.tensor(self.eye, dtype=torch.float32, device=device),
            right=torch.tensor(right_scaled, dtype=torch.float32, device=device),
            up=torch.tensor(up_scaled, dtype=torch.float32, device=device),
            width=self.width,
            height=self.height,
        )

    @property
    def vfov_degrees(self):
        """Vertical field of view in degrees."""
        return math.degrees(2.0 * math.atan(self._up_mag))

    @vfov_degrees.setter
    def vfov_degrees(self, deg):
        self._up_mag = math.tan(math.radians(deg) / 2.0)


# ── GL display via PBO + RegisteredGLBuffer ──────────────────────────────────


class GLDisplay:
    """Manages an RGBA8 GL texture backed by a PBO registered with Warp.

    The compose kernel writes directly into the mapped PBO on the GPU, then
    the PBO contents are copied to the texture via a GPU-side
    ``glTexSubImage2D`` (no CPU round-trip).
    """

    def __init__(self):
        self.texture_id = None
        self.pbo_id = None
        self.registered_buffer = None
        self.width = 0
        self.height = 0

    def ensure_size(self, width, height, device):
        """(Re)allocate the texture, PBO and registered buffer if the size changed."""
        if self.pbo_id is not None and self.width == width and self.height == height:
            return
        self._destroy_gl()

        # --- Texture ---
        self.texture_id = int(GL.glGenTextures(1))
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture_id)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_RGBA8,
            width,
            height,
            0,
            GL.GL_RGBA,
            GL.GL_UNSIGNED_BYTE,
            None,
        )
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

        # --- Pixel Buffer Object ---
        self.pbo_id = int(GL.glGenBuffers(1))
        GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, self.pbo_id)
        GL.glBufferData(
            GL.GL_PIXEL_UNPACK_BUFFER,
            width * height * 4,
            None,
            GL.GL_STREAM_DRAW,
        )
        GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, 0)

        # --- Register PBO with Warp/CUDA ---
        self.registered_buffer = wp.RegisteredGLBuffer(
            self.pbo_id,
            device=None,
            flags=wp.RegisteredGLBuffer.WRITE_DISCARD,
        )

        self.width = width
        self.height = height

    def compose(
        self,
        device,
        color,
        depth,
        normal,
        alpha,
        intersections,
        compose_options,
    ):
        """Run the compose kernel, writing RGBA8 into the mapped PBO."""
        with wp.ScopedDevice(str(device)):
            torch_stream = torch.cuda.current_stream()
            wp_stream = wp.stream_from_torch(torch_stream)
            wp.set_stream(wp_stream)

            buf = self.registered_buffer.map(
                dtype=wp.uint8, shape=(self.height * self.width * 4,)
            )
            wp.launch(
                compose_kernel,
                dim=(self.height, self.width),
                inputs=[
                    color,
                    depth,
                    normal,
                    alpha,
                    intersections,
                    compose_options,
                    buf,
                ],
            )
            wp.synchronize()
            self.registered_buffer.unmap()

    def copy_pbo_to_texture(self):
        """GPU-side copy from PBO into the texture (no CPU involvement)."""
        GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, self.pbo_id)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture_id)
        GL.glTexSubImage2D(
            GL.GL_TEXTURE_2D,
            0,
            0,
            0,
            self.width,
            self.height,
            GL.GL_RGBA,
            GL.GL_UNSIGNED_BYTE,
            ctypes.c_void_p(0),
        )
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, 0)

    def _destroy_gl(self):
        self.registered_buffer = None
        if self.pbo_id is not None:
            GL.glDeleteBuffers(1, [self.pbo_id])
            self.pbo_id = None
        if self.texture_id is not None:
            GL.glDeleteTextures(1, [self.texture_id])
            self.texture_id = None

    def destroy(self):
        self._destroy_gl()
        self.width = 0
        self.height = 0


# ── Viewer ───────────────────────────────────────────────────────────────────


class Viewer:
    def __init__(
        self,
        model,
        camera_init: TorchCamera,
        world_up=None,
        render_mode: str = "rasterize",
    ):
        self.model = model
        self.lock = threading.Lock()
        self.step_count = 0

        self.render_width = 1920
        self.render_height = 1080

        # world_up: numpy array or torch tensor
        if world_up is not None:
            if isinstance(world_up, torch.Tensor):
                world_up = world_up.detach().cpu().numpy()
        self.camera_controller = CameraController(camera_init, world_up=world_up)
        self.camera_controller.width = self.render_width
        self.camera_controller.height = self.render_height

        self.gl_display = GLDisplay()
        self.texture_valid = False

        self.last_render_time = time.time()
        self.render_fps = 0
        self.train_thread = None

        # Training controls
        self.training = False
        self.paused = False
        self.should_step = False
        self.total_iterations = 0

        # Viewer settings
        self.limit_framerate = True
        self.max_framerate = 30

        # Renderer selection (0=Rasterize)
        try:
            self.render_mode = RENDER_MODE_KEYS.index(render_mode)
        except ValueError:
            raise ValueError(
                f"Unknown render_mode {render_mode!r}; expected one of "
                f"{RENDER_MODE_KEYS}."
            )

        # Rasterizer settings
        self.transmittance_threshold = 1e-3
        self.max_intersections = 4096

        # Visualization settings
        self.vis_mode = 0  # 0=RGB, 1=Depth, 2=Normal, 3=Alpha, 4=Intersections
        self.color_map = 1  # 0=Gray, 1=Viridis, 2=Inferno, 3=Turbo
        self.checker_bg = False
        self.bg_color = [0.0, 0.0, 0.0]
        self.max_depth = 25.0
        self.depth_quantile = 0.5

    def step(self, iteration):
        self.step_count = iteration

    def wait_if_paused(self):
        """Call from training thread to block while paused (unless step)."""
        while self.paused and not self.should_step:
            time.sleep(0.01)
        if self.should_step:
            self.should_step = False

    def main_loop(self):
        imgui.style_colors_light()
        io = imgui.get_io()

        # Update Camera
        self.camera_controller.update(io)

        current_time = time.time()
        composed = False

        if self.lock.acquire(blocking=False):
            try:
                device = self.model.points.device

                # Build VisOptions struct from UI state
                vis_options = VisOptions()
                vis_options.transmittance_threshold = self.transmittance_threshold
                vis_options.max_intersections = self.max_intersections
                vis_options.depth_quantile = 1.0 - self.depth_quantile
                if self.vis_mode == 0 and self.checker_bg:
                    vis_options.bkgd_color = wp.vec3f(0.0, 0.0, 0.0)
                else:
                    vis_options.bkgd_color = wp.vec3f(*self.bg_color)

                cam = self.camera_controller.get_eval_camera(device)

                color, depth, normal, alpha, intersections = (
                    self.model.forward_visualization(
                        cam,
                        vis_options=vis_options,
                        render_mode=RENDER_MODE_KEYS[self.render_mode],
                    )
                )

                # Build ComposeOptions
                effective_max_depth = self.max_depth
                compose_opts = ComposeOptions()
                compose_opts.vis_mode = self.vis_mode
                compose_opts.color_map = self.color_map
                compose_opts.effective_max_depth = effective_max_depth
                compose_opts.max_intersections = self.max_intersections
                compose_opts.checker_bg = 1 if self.checker_bg else 0
                compose_opts.bg_color = wp.vec3f(*self.bg_color)
                compose_opts.width = cam.width
                compose_opts.height = cam.height

                # Ensure GL display buffer is allocated for the current size
                self.gl_display.ensure_size(cam.width, cam.height, device)

                # Compose directly into the PBO (GPU-only)
                self.gl_display.compose(
                    device,
                    color,
                    depth,
                    normal,
                    alpha,
                    intersections,
                    compose_opts,
                )

                # PBO -> texture (GPU-to-GPU copy)
                self.gl_display.copy_pbo_to_texture()
                composed = True
            except Exception as e:
                import traceback

                traceback.print_exc()
            finally:
                self.lock.release()

            self.render_fps = 1.0 / (current_time - self.last_render_time + 1e-8)
            self.last_render_time = current_time

        # Frame rate limiting while training is unpaused
        if self.limit_framerate and self.training and not self.paused:
            target = 1.0 / max(self.max_framerate, 1)
            elapsed = time.time() - current_time
            if elapsed < target:
                time.sleep(target - elapsed)

        # -- Controls window ---------------------------------------------------
        imgui.set_next_window_size(imgui.ImVec2(400, 440), imgui.Cond_.first_use_ever)
        imgui.begin("Controls")

        # -- Training ----------------------------------------------------------
        # Spacebar toggles pause
        if self.training and imgui.is_key_pressed(imgui.Key.space, repeat=False):
            self.paused = not self.paused

        if self.training:
            imgui.separator_text("Training")
            imgui.text("Iteration controls: ")
            imgui.same_line()
            if imgui.button(">" if self.paused else "||"):
                self.paused = not self.paused
            if self.paused:
                imgui.same_line()
                disabled = self.should_step
                if disabled:
                    imgui.begin_disabled()
                if imgui.button(">|"):
                    self.should_step = True
                if disabled:
                    imgui.end_disabled()

            if self.total_iterations > 0:
                frac = self.step_count / max(self.total_iterations, 1)
                imgui.progress_bar(frac, imgui.ImVec2(-1, 0))
                imgui.text(f"{self.step_count} / {self.total_iterations}")
            else:
                imgui.text(f"Iteration: {self.step_count}")

        # -- Viewer settings ---------------------------------------------------
        imgui.separator_text("Viewer settings")

        imgui.text(
            f"Resolution: {self.camera_controller.width}x{self.camera_controller.height}"
        )

        fov = self.camera_controller.vfov_degrees
        changed, fov = imgui.slider_float(
            "Field of view",
            fov,
            25.0,
            160.0,
            "%.0f\u00b0",
            imgui.SliderFlags_.logarithmic,
        )
        if changed:
            self.camera_controller.vfov_degrees = fov

        _, self.camera_controller.move_speed = imgui.slider_float(
            "Move speed",
            self.camera_controller.move_speed,
            0.001,
            10.0,
            "%.3f",
            imgui.SliderFlags_.logarithmic | imgui.SliderFlags_.no_round_to_format,
        )

        imgui.text(f"Frame rate: {int(self.render_fps + 0.5)} frames/s")

        if self.training:
            _, self.limit_framerate = imgui.checkbox(
                "Limit frame rate while training", self.limit_framerate
            )
            if self.limit_framerate:
                _, self.max_framerate = imgui.slider_int(
                    "Max frame rate", self.max_framerate, 1, 240
                )

        # -- Renderer settings -------------------------------------------------
        imgui.separator_text("Renderer settings")

        _, self.transmittance_threshold = imgui.slider_float(
            "Weight threshold",
            self.transmittance_threshold,
            1e-3,
            1e0,
            "%.4f",
            imgui.SliderFlags_.logarithmic | imgui.SliderFlags_.no_round_to_format,
        )
        _, self.max_intersections = imgui.slider_int(
            "Max primitives per tile",
            self.max_intersections,
            1,
            4096,
            "%d",
        )

        # -- Visualization settings --------------------------------------------
        imgui.separator_text("Visualization settings")

        # Mode selector
        if imgui.begin_combo("Mode", VIS_MODES[self.vis_mode]):
            for i, name in enumerate(VIS_MODES):
                selected = self.vis_mode == i
                if imgui.selectable(name, selected)[0]:
                    self.vis_mode = i
                if selected:
                    imgui.set_item_default_focus()
            imgui.end_combo()

        # Color map (Depth / Intersections modes)
        if self.vis_mode in (1, 4):
            if imgui.begin_combo("Color map", COLOR_MAPS[self.color_map]):
                for i, name in enumerate(COLOR_MAPS):
                    selected = self.color_map == i
                    if imgui.selectable(name, selected)[0]:
                        self.color_map = i
                    if selected:
                        imgui.set_item_default_focus()
                imgui.end_combo()

        # RGB background options
        if self.vis_mode == 0:
            _, self.checker_bg = imgui.checkbox(
                "Checkerboard background", self.checker_bg
            )
            if not self.checker_bg:
                _, self.bg_color = imgui.color_edit3("Background color", self.bg_color)

        # Depth settings
        if self.vis_mode == 1:
            _, self.max_depth = imgui.slider_float(
                "Max depth",
                self.max_depth,
                1e-5,
                1e3,
                "%.4f",
                imgui.SliderFlags_.logarithmic | imgui.SliderFlags_.no_round_to_format,
            )
            _, self.depth_quantile = imgui.slider_float(
                "Depth quantile",
                self.depth_quantile,
                0.01,
                1.0,
                "%.2f",
            )

        # Help
        imgui.separator()
        imgui.text("WASD to move, Shift/Ctrl up/down")
        imgui.text("Drag to rotate, Space to pause")

        imgui.end()

        # Update render resolution from window size
        disp = io.display_size
        if disp.x > 0 and disp.y > 0:
            new_w = max(8, (int(disp.x) // 8) * 8)
            new_h = max(8, (int(disp.y) // 8) * 8)
            self.camera_controller.width = new_w
            self.camera_controller.height = new_h

        if composed:
            self.texture_valid = True

        # Draw texture as fullscreen background
        if self.texture_valid and self.gl_display.texture_id is not None:
            tex_ref = imgui.ImTextureRef(self.gl_display.texture_id)
            bg = imgui.get_background_draw_list()
            bg.add_image(
                tex_ref,
                imgui.ImVec2(0, 0),
                imgui.ImVec2(disp.x, disp.y),
            )

    def run(self, train_callback=None, total_iterations=0):
        self.total_iterations = total_iterations
        if train_callback is not None:
            self.training = True
            self.paused = True
            self.train_thread = threading.Thread(target=train_callback)
            self.train_thread.daemon = True
            self.train_thread.start()

        params = hello_imgui.RunnerParams()
        params.app_window_params.window_title = "PowerFoam Viewer"
        params.app_window_params.window_geometry.size = (
            self.render_width,
            self.render_height,
        )
        params.imgui_window_params.default_imgui_window_type = (
            hello_imgui.DefaultImGuiWindowType.no_default_window
        )
        params.callbacks.show_gui = self.main_loop
        params.fps_idling.enable_idling = False
        params.ini_folder_type = hello_imgui.IniFolderType.temp_folder

        # Disable vsync and prevent .ini file after the GL context is created
        def _post_init():
            glfw.swap_interval(0)
            imgui.get_io().set_ini_filename("")

        params.callbacks.post_init = _post_init

        hello_imgui.run(params)
