import math
import torch
import torch.nn.functional as F


def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(-1, img1.shape[-1]).mean(0, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse)).mean()


def _to_nchw(img):
    """Convert HWC or NHWC images in [0, 1] to NCHW for ssim/lpips."""
    if img.dim() == 3:
        img = img.unsqueeze(0)
    if img.shape[-1] == 3 and img.shape[1] != 3:
        img = img.permute(0, 3, 1, 2).contiguous()
    return img


def ssim_eval(img1, img2):
    """SSIM for evaluation (HWC inputs in [0, 1] are auto-converted to NCHW)."""
    return ssim(_to_nchw(img1), _to_nchw(img2))


_lpips_model_cache: dict = {}


def lpips_eval(img1, img2, net: str = "vgg"):
    """LPIPS for evaluation. HWC inputs in [0, 1] are auto-converted to NCHW
    in [-1, 1]. The underlying network is lazily loaded once per device/net.
    """
    import lpips as _lpips

    img1 = _to_nchw(img1.float())
    img2 = _to_nchw(img2.float())
    device = img1.device

    key = (net, str(device))
    model = _lpips_model_cache.get(key)
    if model is None:
        model = _lpips.LPIPS(net=net, verbose=False).to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        _lpips_model_cache[key] = model

    img1_n = img1.clamp(0.0, 1.0) * 2.0 - 1.0
    img2_n = img2.clamp(0.0, 1.0) * 2.0 - 1.0
    with torch.no_grad():
        return model(img1_n, img2_n).mean()


@torch.jit.script
def gaussian(window_size: int, sigma: float) -> torch.Tensor:
    gauss = torch.tensor(
        [
            math.exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ],
        dtype=torch.float32,
    )
    return gauss / gauss.sum()


@torch.jit.script
def create_window(window_size: int, channel: int) -> torch.Tensor:
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


@torch.jit.script
def _ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window: torch.Tensor,
    window_size: int,
    channel: int,
    size_average: bool = True,
) -> torch.Tensor:
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


_ssim_window_cache: dict = {}


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    key = (window_size, channel, img1.device, img1.dtype)

    window = _ssim_window_cache.get(key)
    if window is None:
        window = create_window(window_size, channel)
        if img1.is_cuda:
            window = window.to(device=img1.device, dtype=img1.dtype)
        else:
            window = window.type_as(img1)
        _ssim_window_cache[key] = window

    return _ssim(img1, img2, window, window_size, channel, size_average)
