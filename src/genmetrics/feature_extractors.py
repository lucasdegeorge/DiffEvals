"""Implements all the necessary feature extractors."""

import torch
import transformers
import os
import numpy as np
from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
from torch import Tensor
from typing import List, Optional
import PIL

class NoTrainInceptionV3(FeatureExtractorInceptionV3):
    def __init__(
        self,
        name: List[str],
        features_list: str,
        feature_extractor_weights_path: Optional[str] = None,
    ) -> None:
        super().__init__(name, features_list, feature_extractor_weights_path)
        # put into evaluation mode
        self.eval()

    def train(self, mode: bool) -> "NoTrainInceptionV3":
        """the inception network should not be able to be switched away from evaluation mode."""
        return super().train(False)

    def forward(self, x: Tensor) -> Tensor:
        out = super().forward(x)
        return out
    
class CLIPFeatureExtractor(torch.nn.Module):
    def __init__(self, path="openai/clip-vit-large-patch14"):
        super().__init__()
        self.processor = transformers.CLIPImageProcessor.from_pretrained(path)
        self.model = transformers.CLIPVisionModel.from_pretrained(path)
        self.model.eval().requires_grad_(False)

    def forward(self, images):
        images = self.processor(images=images, return_tensors="pt").to(
            self.model.device
        )
        features = self.model(**images).last_hidden_state[:, 0, :]
        return features


class Dinov2FeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        os.environ["XFORMERS_DISABLED"] = "1"
        self.model = torch.hub.load(
            "facebookresearch/dinov2:main",
            "dinov2_vitl14",
            trust_repo=True,
            verbose=False,
            skip_validation=True,
        )
        self.model.eval().requires_grad_(False)

    def forward(self, images):
        device = images.device
        x = images.to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        x = np.stack(
            [
                np.uint8(
                    PIL.Image.fromarray(xx, "RGB").resize(
                        (224, 224), PIL.Image.Resampling.BICUBIC
                    )
                )
                for xx in x
            ]
        )
        x = torch.from_numpy(x).permute(0, 3, 1, 2).to(device)
        x = x.to(torch.float32) / 255
        x = x - torch.tensor([0.485, 0.456, 0.406], device=device).reshape(1, -1, 1, 1)
        x = x / torch.tensor([0.229, 0.224, 0.225], device=x.device).reshape(
            1, -1, 1, 1
        )
        features = self.model(x)
        return features