import numpy as np
from scipy.special import softmax, log_softmax
from scipy import linalg
import torch
from torch.utils.data import Dataset
import os
from PIL import Image
from glob import glob
from prdc import compute_prdc
import json


## Defining the calculation functions ##
def calculate_fid(
    real_features, fake_features, real_mu_sigma=(None, None), fake_mu_sigma=(None, None)
):
    """Calculates the FID score between the real and fake features."""

    ## Calculating the mean and covariance of real image features ##
    if real_mu_sigma != (None, None):
        mean_real = real_mu_sigma[0].detach().cpu().numpy()
        sigma_real = real_mu_sigma[1].detach().cpu().numpy()

    else:
        ## Making the tensors in double precision ##
        real_features = real_features.double().detach().cpu().numpy()
        mean_real = np.mean(real_features, axis=0)
        sigma_real = np.cov(real_features, rowvar=False)

    ## Calculating the mean and covariance of fake image features ##
    if fake_mu_sigma != (None, None):
        mean_fake = fake_mu_sigma[0].detach().cpu().numpy()
        sigma_fake = fake_mu_sigma[1].detach().cpu().numpy()
    else:
        ## Making the tensors in double precision ##
        fake_features = fake_features.double().detach().cpu().numpy()
        mean_fake = np.mean(fake_features, axis=0)
        sigma_fake = np.cov(fake_features, rowvar=False)

    ## Calculating the FID score ##
    m = np.square(mean_fake - mean_real).sum()
    s, _ = linalg.sqrtm(sigma_fake.dot(sigma_real), disp=False)
    value = float(np.real(m + np.trace(sigma_fake + sigma_real - 2 * s)))

    return value


def calculate_is(logits, splits=10):
    "Calculates the IS score of the fake features."

    ## Randomizing the features ##
    torch.manual_seed(2020)
    torch.cuda.manual_seed(2020)
    logits = logits.double()
    logits = logits[torch.randperm(logits.size(0))]

    ## Getting the probabilities ##
    conditional_prob_yx = logits.softmax(dim=1)
    log_conditional_prob_yx = logits.log_softmax(dim=1)

    ## Splitting the conditional probs for final calculation ##
    conditional_prob_yx = conditional_prob_yx.chunk(splits, dim=0)
    log_conditional_prob_yx = log_conditional_prob_yx.chunk(splits, dim=0)

    ## Calculating the marginal probability ##
    log_marginal_prob_y = torch.stack(
        [torch.log(prob.mean(dim=0, keepdim=True)) for prob in conditional_prob_yx]
    )

    kl = [
        p_c * (log_p_c - log_p_m)
        for p_c, log_p_c, log_p_m in zip(
            conditional_prob_yx, log_conditional_prob_yx, log_marginal_prob_y
        )
    ]

    kl = torch.stack([k.sum(axis=1).mean().exp() for k in kl])

    return kl.mean().item()


def calculate_prdc(real_features, fake_features, k, prdc_splits=5):
    """Calculates the PRDC score between the real and fake features."""

    real_features = real_features.double().detach().cpu()
    fake_features = fake_features.double().detach().cpu()

    real_features = real_features.chunk(prdc_splits, dim=0)
    fake_features = fake_features.chunk(prdc_splits, dim=0)

    precision = []
    recall = []
    density = []
    coverage = []

    for i in range(prdc_splits):
        real = real_features[i].numpy()
        fake = fake_features[i].numpy()

        metrics = compute_prdc(real_features=real, fake_features=fake, nearest_k=k)
        precision.append(metrics["precision"])
        recall.append(metrics["recall"])
        density.append(metrics["density"])
        coverage.append(metrics["coverage"])

    return np.mean(precision), np.mean(recall), np.mean(density), np.mean(coverage)


def open_txt_file(path: str):
    """Open a txt file and return the contents."""

    with open(path, "r") as fp:
        contents = fp.read().splitlines()
        return contents


def open_jsonl_file(path: str):
    """Open a json file and return the contents."""

    with open(path, "r") as fp:
        contents = [json.loads(line) for line in fp]
        return contents


def open_json_file(path: str):
    """Open a json file and return the contents."""

    with open(path, "r") as fp:
        contents = json.load(fp)
        return contents

## Making dataset ##
class GenDataset(Dataset):
    """Implements the dataset."""

    def __init__(self, image_path=None, ref_caption_path=None, gen_caption_path=None):
        super().__init__()

        assert (image_path is not None) or (
            ref_caption_path is not None
        ), "Need at least one of image_path or ref_caption_path."

        self.image_path = image_path
        self.ref_caption_path = ref_caption_path
        self.gen_caption_path = (
            gen_caption_path if gen_caption_path is not None else ref_caption_path
        )
        self.mean = torch.tensor([False], dtype=torch.bool)
        self.sigma = torch.tensor([False], dtype=torch.bool)

        ## Getting the images if image path is provided ##
        if image_path is not None:
            if ".npz" in image_path:
                with np.load(image_path) as file:
                    self.images = torch.from_numpy(file["arr_0"]).permute(0, 3, 1, 2)
                    try:
                        self.mean = torch.from_numpy(file["mu"])
                        self.sigma = torch.from_numpy(file["sigma"])
                    except KeyError:
                        pass

            elif os.path.isdir(image_path):
                self.images = [
                    torch.from_numpy(np.array(Image.open(file).convert("RGB"))).permute(
                        2, 0, 1
                    )
                    for file in sorted(glob(f"{image_path}/*.png"))
                    + sorted(glob(f"{image_path}/*.jpg"))
                ]

                self.images = torch.stack(self.images, dim=0)

            else:
                raise ValueError("Invalid image path.")

        if ref_caption_path is not None:
            ## Getting the reference captions ##
            self.ref_captions = self._get_captions(ref_caption_path)
            self.gen_captions = self._get_captions(gen_caption_path)

    def _get_captions(self, path):
        """Helper function to get captions from a file."""
        if path.endswith(".txt"):
            all_contents = open_txt_file(path)
            captions = [caption.lstrip('b"').rstrip('"') for caption in all_contents]
            return captions
        elif path.endswith(".jsonl"):
            all_contents = open_jsonl_file(path)
            captions = [
                content["prompt"].lstrip('b"').rstrip('"') for content in all_contents
            ]
            return captions
        elif path.endswith(".json"):
            all_contents = open_json_file(path)
            captions = all_contents["captions"]
            return captions
        else:
            raise ValueError("Unsupported caption file format.")

    def __getitem__(self, idx):
        batch = {
            "mean": self.mean,
            "sigma": self.sigma,
        }
        if self.image_path is not None:
            batch.update({"image": self.images[idx]})
        if self.ref_caption_path is not None:
            batch.update({"ref_caption": self.ref_captions[idx]})
        if self.gen_caption_path is not None:
            batch.update({"gen_caption": self.gen_captions[idx]})
        return batch

    def __len__(self):
        return (
            len(self.images) if self.image_path is not None else len(self.ref_captions)
        )