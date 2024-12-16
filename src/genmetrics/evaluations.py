"""Implements evaluation metrics for the diffusion models.

Majorly, this file implements the following evaluation metrics:
    1. FID
    2. IS
    3. Precision
    4. Recall
    5. Density
    6. Coverage
    7. ClipScore
"""

import torch
import torch.nn as nn
from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat
from torchmetrics.multimodal.clip_score import CLIPScore
from tqdm import tqdm
from typing import List

from .utils import (
    calculate_fid,
    calculate_is,
    calculate_prdc,
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
        metrics: List[str] = ["inception", "clip"],
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
            metric in ["inception", "clip"] for metric in metrics
        ), "Sorry, as of now only inception and clip is supported!"

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
                self.__getattribute__(f"{prefix}_features_{name}").append(preds[0])

                if real == False:
                    self.__getattribute__(f"{prefix}_logits_{name}").append(preds[1])
            else:
                self.__getattribute__(f"{prefix}_features_{name}").append(preds)

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