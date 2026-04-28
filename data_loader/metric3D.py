import os

import numpy as np
from PIL import Image
import torch
import tqdm


class Metric3DEstimator:
    def __init__(self):
        self.model = torch.hub.load(
            "yvanyin/metric3d", "metric3d_vit_large", pretrain=True
        )
        self.model.cuda().eval()

    def process_dir(self, input_files, output_dir):
        for path in tqdm.tqdm(input_files):
            image = Image.open(path).convert("RGB")
            depth, normal, confidence = self.estimate(image)
            image.close()
            data = {
                "depth": depth,
                "normal": normal,
                "confidence": confidence,
            }

            name = os.path.split(path)[1]
            filename = os.path.splitext(name)[0]
            torch.save(data, os.path.join(output_dir, f"{filename}.pt"))

    def estimate(self, image):
        input_size = (616, 1064)
        input_aspect = input_size[1] / input_size[0]
        image_aspect = image.width / image.height

        if input_aspect < image_aspect:
            new_width = input_size[1]
            new_height = int(input_size[1] / image.width * image.height)
            image = image.resize((new_width, new_height), resample=Image.LANCZOS)
            total_padding = input_size[0] - new_height
            top_padding = total_padding // 2
            bottom_padding = total_padding - top_padding
            left_padding = 0
            right_padding = 0
        else:
            new_height = input_size[0]
            new_width = int(input_size[0] / image.height * image.width)
            image = image.resize((new_width, new_height), resample=Image.LANCZOS)
            total_padding = input_size[1] - new_width
            left_padding = total_padding // 2
            right_padding = total_padding - left_padding
            top_padding = 0
            bottom_padding = 0

        image = torch.tensor(np.array(image)).float()
        mean = torch.tensor([123.675, 116.28, 103.53]).float()
        std = torch.tensor([58.395, 57.12, 57.375]).float()
        image = (image - mean) / std
        image = image.permute(2, 0, 1)[None]

        image = torch.nn.functional.pad(
            image,
            (left_padding, right_padding, top_padding, bottom_padding),
        )

        with torch.no_grad():
            depth, confidence, output_dict = self.model.inference(
                {"input": image.cuda()}
            )
            normals = output_dict["prediction_normal"]
            depth = depth.cpu()[
                0,
                0,
                top_padding : input_size[0] - bottom_padding,
                left_padding : input_size[1] - right_padding,
            ]
            normals = normals.cpu()[
                0,
                0:3,
                top_padding : input_size[0] - bottom_padding,
                left_padding : input_size[1] - right_padding,
            ]
            confidence = confidence.cpu()[
                0,
                0,
                top_padding : input_size[0] - bottom_padding,
                left_padding : input_size[1] - right_padding,
            ]

            return depth, normals, confidence
