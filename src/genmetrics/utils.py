import numpy as np
from scipy import linalg
import torch
from torch.utils.data import Dataset
import os
from PIL import Image
from glob import glob
from prdc import compute_prdc


## Defining the calculation functions ##
def calculate_fid(real_features, fake_features):
    """Calculates the FID score between the real and fake features."""

    ## Making the tensors in double precision ##
    real_features = real_features.double().detach().cpu().numpy()
    fake_features = fake_features.double().detach().cpu().numpy()

    ## Calculating the mean and covariance ##
    mean_real = np.mean(real_features, axis=0)
    mean_fake = np.mean(fake_features, axis=0)

    sigma_real = np.cov(real_features, rowvar=False)
    sigma_fake = np.cov(fake_features, rowvar=False)

    ## Calculating the FID score ##
    ## Code taken from https://github.com/openai/guided-diffusion/blob/main/evaluations/evaluator.py ##
    mean_diff = mean_real - mean_fake
    covmean, _ = linalg.sqrtm(sigma_real.dot(sigma_fake), disp=False)

    if not np.isfinite(covmean).all():
        offset = np.eye(sigma_real.shape[0]) * 1e-6
        covmean = linalg.sqrtm((sigma_real + offset).dot(sigma_fake + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    fid = (
        mean_diff.dot(mean_diff)
        + np.trace(sigma_real)
        + np.trace(sigma_fake)
        - 2 * tr_covmean
    )

    return fid


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

    print(len(kl))

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


## Making dataset ##
class GenDataset(Dataset):
    """Implements the dataset."""

    def __init__(self, path):
        super().__init__()

        ## Getting the data ##
        if ".npz" in path:
            with np.load(path) as file:
                self.images = torch.from_numpy(file["arr_0"]).permute(0, 3, 1, 2)

        elif os.path.isdir(path):
            self.images = [
                torch.from_numpy(np.array(Image.open(file).convert("RGB"))).permute(
                    2, 0, 1
                )
                for file in sorted(glob(f"{path}/*.png"))
            ]

            self.images = torch.stack(self.images, dim=0)

        else:
            raise ValueError("Invalid path.")

    def __getitem__(self, idx):
        return self.images[idx]

    def __len__(self):
        return len(self.images)
