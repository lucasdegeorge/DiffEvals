"""Implements evaluation metrics for the diffusion models.

Majorly, this file implements the following evaluation metrics:
    1. FID
    2. IS
    3. Precision
    4. Recall
    5. Density
    6. Coverage
    7. ClipScore
    8. JinaClipScore
    9. PickScore
    10. Aesthetic Score
    11. HPSv2 Score
    12. ImageReward Score
"""

import torch
import torch.nn as nn
from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat
from torchmetrics.multimodal.clip_score import CLIPScore
from torchvision import transforms
from tqdm import tqdm
from typing import List
from os.path import expanduser
from urllib.request import urlretrieve
import os

from .utils import (
    calculate_fid,
    calculate_is,
    calculate_prdc,
    GenDataset,
)

from .feature_extractors import (
    NoTrainInceptionV3,
    CLIPFeatureExtractor,
    Dinov2FeatureExtractor,
)


class GenMetric(Metric):
    """Implements the evaluation metrics for the generator model."""

    def __init__(
        self,
        metrics: List[str] = [
            "inception",
            "clip",
            "jina_clip",
            "pickscore",
            "aesthetic_score",
            "hpsv2_score",
            "image_reward_score",
        ],
        feature_extractors_for_inception_metrics: List[str] = [
            "inceptionv3",
            "clip",
            "dinov2",
        ],
        **kwargs,
    ):
        """Constructor.
        params: metrics: The evaluation metrics to be calculated.
        params: feature_extractors_for_inception_metrics: The feature extractors to be used for the inception metrics.
        """

        super().__init__(**kwargs)

        ## Checking the validity of the metrics ##
        assert all(
            metric
            in [
                "inception",
                "clip",
                "jina_clip",
                "pickscore",
                "aesthetic_score",
                "hpsv2_score",
                "image_reward_score",
            ]
            for metric in metrics
        ), f"Sorry, one of the requested metric is not implemented!"

        if "inception" in metrics:
            assert (
                feature_extractors_for_inception_metrics is not None
            ), "Feature extractors for inception metrics must be provided."

        ## Setting the attributes ##
        self.metrics = metrics

        ## Setting up the metric subclasses ##
        ## 1. Inception Metrics ##
        if "inception" in metrics:
            feature_extractors_dict = {}
            for name in feature_extractors_for_inception_metrics:
                if name == "inceptionv3":
                    feature_extractors_dict[name] = NoTrainInceptionV3(
                        name=name,
                        features_list=["2048", "logits_unbiased"],
                    )
                if name == "clip":
                    feature_extractors_dict[name] = CLIPFeatureExtractor(
                        path="openai/clip-vit-large-patch14"
                    )
                if name == "dinov2":
                    feature_extractors_dict[name] = Dinov2FeatureExtractor()

            feature_extractors = nn.ModuleDict(feature_extractors_dict)

            self.inception_metrics = InceptionMetric(
                features=feature_extractors,
                which_scores=["fid", "is", "prdc"],
            )

        ## 2. CLIPScore ##
        if "clip" in metrics:
            self.clip_score_metrics = CLIPScore("openai/clip-vit-large-patch14")

        if "jina_clip" in metrics:
            self.jina_clip_score_metrics = CLIPJinaScore()

        ## 3. PickScore ##
        if "pickscore" in metrics:
            self.pick_score_metrics = PickScore()

        ## 4. Aesthetic Score ##
        if "aesthetic_score" in metrics:
            self.aesthetic_score_metrics = AestheticScore()

        ## 5. HPSv2 Score ##
        if "hpsv2_score" in metrics:
            self.hpsv2_score_metrics = HPSv2Score()

        ## 6. ImageReward Score ##
        if "image_reward_score" in metrics:
            self.image_reward_score_metrics = ImageRewardScore()

    def update(
        self,
        images,
        mean: torch.tensor = torch.tensor([False], dtype=torch.bool),
        sigma: torch.tensor = torch.tensor([False], dtype=torch.bool),
        real: bool = True,
        captions=None,
    ):
        """Updates the states of the metrics."""

        if "inception" in self.metrics:
            self.inception_metrics.update(images, mean=mean, sigma=sigma, real=real)

        if "clip" in self.metrics:
            if real == False:
                assert captions is not None, "Captions must be provided for CLIPScore."
                self.clip_score_metrics.update(images, captions)

        if "jina_clip" in self.metrics:
            if real == False:
                assert (
                    captions is not None
                ), "Captions must be provided for JinaCLIPScore."
                self.jina_clip_score_metrics.update(images, captions)

        if "pickscore" in self.metrics:
            if real == False:
                assert captions is not None, "Captions must be provided for PickScore."
                self.pick_score_metrics.update(images, captions)

        if "aesthetic_score" in self.metrics:
            if real == False:
                self.aesthetic_score_metrics.update(images)

        if "hpsv2_score" in self.metrics:
            if real == False:
                assert captions is not None, "Captions must be provided for HPSv2Score."
                self.hpsv2_score_metrics.update(images, captions)

        if "image_reward_score" in self.metrics:
            if real == False:
                assert captions is not None, "Captions must be provided for ImageRewardScore."
                self.image_reward_score_metrics.update(images, captions)

    def compute(self):
        """Computes the evaluation metrics."""
        eval_scores = {}

        print("Computing the evaluation metrics...")
        if "inception" in self.metrics:
            print("Computing the inception metrics...")
            eval_scores.update(self.inception_metrics.compute())

        if "clip" in self.metrics:
            print("Computing the clip score...", end="")
            eval_scores["clip_score"] = self.clip_score_metrics.compute().item()
            print("Done!")

        if "jina_clip" in self.metrics:
            print("Computing the Jina-CLIP score...", end="")
            eval_scores["jina_clip_score"] = (
                self.jina_clip_score_metrics.compute().item()
            )
            print("Done!")

        if "pickscore" in self.metrics:
            print("Computing the PickScore...", end="")
            eval_scores["pick_score"] = self.pick_score_metrics.compute().item()
            print("Done!")

        if "aesthetic_score" in self.metrics:
            print("Computing the Aesthetic Score...", end="")
            eval_scores["aesthetic_score"] = (
                self.aesthetic_score_metrics.compute().item()
            )
            print("Done!")

        if "hpsv2_score" in self.metrics:
            print("Computing the HPSv2 Score...", end="")
            eval_scores["hpsv2_score"] = self.hpsv2_score_metrics.compute().item()
            print("Done!")

        if "image_reward_score" in self.metrics:
            print("Computing the ImageReward Score...", end="")
            eval_scores["image_reward_score"] = (
                self.image_reward_score_metrics.compute().item()
            )
            print("Done!")

        return eval_scores


## Inception Metric ##


class InceptionMetric(Metric):
    """Implements the inception evaluation metrics. This is a group of metrics:
    1. FID
    2. IS
    3. Precision
    4. Recall
    5. Density
    6. Coverage
    """

    def __init__(
        self,
        features,
        which_scores: list,
        is_score_splits: int = 10,
        k: int = 3,
        prdc_splits: int = 5,
        **kwargs,
    ):
        """Constructor.

        params: features: The features to be extracted from the generator model.
        params: which_scores: Which evaluation metrics to be calculated.
        params: is_score_splits: The number of splits for the IS score.
        params: k: The k value for the k-nearest distance calculation.
        """
        super().__init__(**kwargs)

        ## Setting the attributes ##
        self.which_scores = which_scores
        self.is_score_splits = is_score_splits
        self.k = k
        self.prdc_splits = prdc_splits
        self.feature_extractor_names = list(features.keys())
        self.feature_extractors = nn.ModuleDict(features)

        if "is" in which_scores:
            assert any(
                "inception" in name for name in self.feature_extractor_names
            ), f"IS can only be calculated for inceptionv3 feature extractors."

        ## Setting the states ##
        for feature in self.feature_extractor_names:
            self.add_state(f"real_features_{feature}", default=[], dist_reduce_fx="cat")
            self.add_state(f"fake_features_{feature}", default=[], dist_reduce_fx="cat")
            self.add_state(f"fake_logits_{feature}", default=[], dist_reduce_fx="cat")
            # if "incpetion" in feature:
            self.__setattr__(f"real_mean_{feature}", None)
            self.__setattr__(f"real_sigma_{feature}", None)
            self.__setattr__(f"fake_mean_{feature}", None)
            self.__setattr__(f"fake_sigma_{feature}", None)

    # def update(self,)
    def update(
        self, images, mean: torch.Tensor, sigma: torch.Tensor, real: bool = True
    ):
        """Updates the states of the metrics."""

        ## Setting the prefix ##
        prefix = "real" if real else "fake"

        # ## Extracting the features ##
        for name in self.feature_extractor_names:
            if "inception" in name:
                ## Setting the mean and sigma ##
                self.__setattr__(
                    f"{prefix}_mean_{name}",
                    mean[0] if mean.dtype != torch.bool else None,
                )
                self.__setattr__(
                    f"{prefix}_sigma_{name}",
                    sigma[0] if mean.dtype != torch.bool else None,
                )

            preds = self.feature_extractors[name](images)

            if isinstance(preds, tuple):
                self.__getattribute__(f"{prefix}_features_{name}").append(
                    preds[0].detach().cpu()
                )

                if real == False:
                    self.__getattribute__(f"{prefix}_logits_{name}").append(
                        preds[1].detach().cpu()
                    )
            else:
                self.__getattribute__(f"{prefix}_features_{name}").append(
                    preds.detach().cpu()
                )

    def compute(self):
        """Computes the evaluation metrics."""
        eval_scores = {}

        ## Getting the features and doing dim zero cat ##
        for name in self.feature_extractor_names:
            real_features = dim_zero_cat(self.__getattribute__(f"real_features_{name}"))
            fake_features = dim_zero_cat(self.__getattribute__(f"fake_features_{name}"))

            if "fid" in self.which_scores:
                print(f"Calculating FID for {name}...", end="")
                eval_scores[f"fid_{name}"] = calculate_fid(
                    real_features,
                    fake_features,
                    real_mu_sigma=(
                        self.__getattribute__(f"real_mean_{name}"),
                        self.__getattribute__(f"real_sigma_{name}"),
                    ),
                    fake_mu_sigma=(
                        self.__getattribute__(f"fake_mean_{name}"),
                        self.__getattribute__(f"fake_sigma_{name}"),
                    ),
                )
                print("Done!")

            if "is" in self.which_scores and "inception" in name:
                print(f"Calculating IS for {name}...", end="")
                fake_logits = dim_zero_cat(self.__getattribute__(f"fake_logits_{name}"))
                eval_scores[f"is_{name}"] = calculate_is(
                    fake_logits, splits=self.is_score_splits
                )
                print("Done!")

            if "prdc" in self.which_scores:
                print(f"Calculating PRDC for {name}. It is done in batches...")
                precision, recall, density, coverage = calculate_prdc(
                    real_features=real_features,
                    fake_features=fake_features,
                    k=self.k,
                    prdc_splits=self.prdc_splits,
                )

                eval_scores[f"precision_{name}"] = precision
                eval_scores[f"recall_{name}"] = recall
                eval_scores[f"density_{name}"] = density
                eval_scores[f"coverage_{name}"] = coverage
                print("Done!")

        return eval_scores


class CLIPJinaScore(Metric):
    """Implements the CLIPScore using the Jina-CLIP-v2 model."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model, self.processor = self._get_jina_model_and_processor()
        self.add_state("score", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state(
            "n_samples", torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum"
        )

    def update(self, images, text):
        """Update score on a batch of images and text."""
        score, n_samples = self._score_update(images, text, self.model, self.processor)
        self.score += score.sum(0)
        self.n_samples += n_samples

    def compute(self):
        """Compute accumulated score."""
        return torch.max(self.score / self.n_samples, torch.zeros_like(self.score))

    def _get_jina_model_and_processor(self):
        """Returns the Jina-CLIP-v2 model and processor."""
        from transformers import AutoModel, AutoProcessor

        model = AutoModel.from_pretrained("jinaai/jina-clip-v2", trust_remote_code=True)

        processor = AutoProcessor.from_pretrained(
            "jinaai/jina-clip-v2", trust_remote_code=True
        )

        return model, processor

    def _score_update(self, images, text, model, processor):
        """Update score on a batch of images and text."""

        device = images[0].device

        processed_input = processor(
            text=text,
            images=[transforms.functional.to_pil_image(i.cpu()) for i in images],
            return_tensors="pt",
            padding=True,
        )

        img_features = model.get_image_features(
            processed_input["pixel_values"].to(device)
        )
        img_features = img_features / img_features.norm(p=2, dim=-1, keepdim=True)

        txt_features = model.get_text_features(
            processed_input["input_ids"].to(device),
            processed_input["attention_mask"].to(device),
        )
        txt_features = txt_features / txt_features.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity between feature vectors
        score = 100 * (img_features * txt_features).sum(axis=-1)
        return score, len(text)


class PickScore(Metric):
    """Implements the Pickscore using laion clip model."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model, self.processor = self._get_pickscore_model_and_processor()
        self.add_state("score", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state(
            "n_samples", torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum"
        )

    def update(self, images, text):
        """Update score on a batch of images and text."""
        score, n_samples = self._score_update(images, text, self.model, self.processor)
        self.score += score.sum(0)
        self.n_samples += n_samples

    def compute(self):
        """Compute accumulated score."""
        return self.score / self.n_samples

    def _get_pickscore_model_and_processor(self):
        """Returns the pickscore model and processor."""
        from transformers import AutoModel, AutoProcessor

        model = AutoModel.from_pretrained(
            "yuvalkirstain/PickScore_v1", trust_remote_code=True
        )

        processor = AutoProcessor.from_pretrained(
            "laion/CLIP-ViT-H-14-laion2B-s32B-b79K", trust_remote_code=True
        )

        return model.eval(), processor

    def _score_update(self, images, text, model, processor):
        """Update score on a batch of images and text."""

        device = images[0].device

        processed_input = processor(
            text=text,
            images=[transforms.functional.to_pil_image(i.cpu()) for i in images],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )

        img_features = model.get_image_features(
            processed_input["pixel_values"].to(device)
        )
        img_features = img_features / img_features.norm(p=2, dim=-1, keepdim=True)

        txt_features = model.get_text_features(
            processed_input["input_ids"].to(device),
            processed_input["attention_mask"].to(device),
        )
        txt_features = txt_features / txt_features.norm(p=2, dim=-1, keepdim=True)

        score = model.logit_scale.exp() * torch.diag(txt_features @ img_features.T)
        
        return score, len(text)


class AestheticScore(Metric):
    """Implements the Aesthetics score using clip model."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.aesthetic_predictor, self.feature_extractor, self.preprocess = (
            self._get_aesthetic_score_models_and_processor()
        )
        self.add_state("score", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state(
            "n_samples", torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum"
        )

    def update(self, images):
        """Update score on a batch of images and text."""
        score, n_samples = self._score_update(
            images, self.aesthetic_predictor, self.feature_extractor, self.preprocess
        )
        self.score += score.sum()
        self.n_samples += n_samples

    def compute(self):
        """Compute accumulated score."""
        return self.score / self.n_samples

    def _get_aesthetic_score_models_and_processor(self):
        """Returns the aesthetic models and processor."""
        import open_clip

        ## 1. We will load the aesthetic prediction head ##
        home = expanduser("~")
        cache_folder = home + "/.cache/emb_reader"
        path_to_model = cache_folder + "/sa_0_4_vit_l_14_linear.pth"
        if not os.path.exists(path_to_model):
            os.makedirs(cache_folder, exist_ok=True)
            url_model = "https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_vit_l_14_linear.pth?raw=true"
            urlretrieve(url_model, path_to_model)
        aesthetic_predictor = nn.Linear(768, 1)
        aesthetic_predictor.load_state_dict(torch.load(path_to_model))
        aesthetic_predictor.eval()

        ## 2. Then the clip model and the processor ##
        feature_extractor, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="openai"
        )

        return aesthetic_predictor.eval(), feature_extractor.eval(), preprocess

    def _score_update(self, images, aesthetic_predictor, feature_extractor, preprocess):
        """Update score on a batch of images."""

        device = images[0].device

        images = torch.cat(
            [
                preprocess(transforms.functional.to_pil_image(i.cpu()))
                .unsqueeze(0)
                .to(device)
                for i in images
            ],
            dim=0,
        )

        image_features = feature_extractor.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        predictions = aesthetic_predictor(image_features)

        score = predictions.detach().cpu()
        return score, len(predictions)


class HPSv2Score(Metric):
    """Implements the Aesthetics score using clip model."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model, self.preprocess, self.tokenizer = (
            self._get_hpsv2score_models_and_processor()
        )
        self.add_state("score", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state(
            "n_samples", torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum"
        )

    def update(self, images, texts):
        """Update score on a batch of images and text."""
        score, n_samples = self._score_update(
            images, texts, self.model, self.preprocess, self.tokenizer
        )
        self.score += score.sum()
        self.n_samples += n_samples

    def compute(self):
        """Compute accumulated score."""
        return self.score / self.n_samples

    def _get_hpsv2score_models_and_processor(self):
        """Returns the hpsv2 models and processor."""
        import huggingface_hub
        from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
        from hpsv2.utils import root_path, hps_version_map

        model, _, preprocess = create_model_and_transforms(
            "ViT-H-14",
            "laion2B-s32B-b79K",
            precision="amp",
            device=self.device,
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=False,
            force_image_size=None,
            pretrained_image=False,
            image_mean=None,
            image_std=None,
            light_augmentation=True,
            aug_cfg={},
            output_dict=True,
            with_score_predictor=False,
            with_region_predictor=False,
        )

        cp = huggingface_hub.hf_hub_download("xswu/HPSv2", hps_version_map["v2.1"])
        checkpoint = torch.load(cp, map_location=self.device)
        model.load_state_dict(checkpoint["state_dict"])
        tokenizer = get_tokenizer("ViT-H-14")

        return model.eval(), preprocess, tokenizer

    def _score_update(self, images, texts, model, preprocess, tokenizer):
        """Update score on a batch of images."""

        device = images[0].device

        images = torch.cat(
            [
                preprocess(transforms.functional.to_pil_image(i.cpu()))
                .unsqueeze(0)
                .to(device)
                for i in images
            ],
            dim=0,
        )

        texts = tokenizer(texts).to(device)

        outputs = model(images, texts)
        image_features, text_features = (
            outputs["image_features"],
            outputs["text_features"],
        )
        logits_per_image = image_features @ text_features.T
        score = torch.diag(logits_per_image).detach().cpu()

        return score, len(texts)


class ImageRewardScore(Metric):
    """Implements the Imagereward score."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model, self.preprocess, self.mlp, self.mean, self.std = (
            self._get_image_reward_score_model()
        )
        self.add_state("score", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state(
            "n_samples", torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum"
        )

    def update(self, images, texts):
        """Update score on a batch of images and text."""
        score, n_samples = self._score_update(images, texts, self.model)
        self.score += score.sum()
        self.n_samples += n_samples

    def compute(self):
        """Compute accumulated score."""
        return self.score / self.n_samples

    def _get_image_reward_score_model(self):
        """Returns the image reward models."""
        import ImageReward as RM

        base = RM.load("ImageReward-v1.0")

        return base.blip.eval(), base.preprocess, base.mlp, base.mean, base.std

    def _score_update(self, images, texts, model):
        """Update score on a batch of images."""

        device = images[0].device

        images = torch.cat(
            [
                self.preprocess(transforms.functional.to_pil_image(i.cpu()))
                .unsqueeze(0)
                .to(device)
                for i in images
            ],
            dim=0,
        )

        # text encode
        text_input = model.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=35,
            return_tensors="pt",
        ).to(device)

        # image encode
        image_embeds = model.visual_encoder(images)

        # text encode cross attention with image
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            device
        )
        text_output = model.text_encoder(
            text_input.input_ids,
            attention_mask=text_input.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        txt_features = text_output.last_hidden_state[
            :, 0, :
        ].float()  # (feature_dim)
        score = self.mlp(txt_features)
        score = (score - self.mean) / self.std

        return score.detach().cpu(), len(texts)
    