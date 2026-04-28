import torch
import warp as wp


@torch.jit.ignore
def launch_kernel_from_torch(device, kernel, dim, inputs, block_dim):
    torch_stream = torch.cuda.current_stream()
    wp_stream = wp.stream_from_torch(torch_stream)

    for input in inputs:
        if isinstance(input, torch.Tensor):
            if input.device.type == "cuda":
                input = input.contiguous()
            else:
                raise ValueError("Only CUDA tensors are supported as inputs")

    wp.launch(
        kernel,
        dim=dim,
        inputs=inputs,
        block_dim=block_dim,
        device=str(device),
        stream=wp_stream,
    )
