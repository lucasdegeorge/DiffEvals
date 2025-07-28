"""Implementation of a simple example usage of the module."""

from tqdm import tqdm
from torch.utils.data import DataLoader
from genmetrics import GenMetric, GenDataset


def main():
    ## Initialize the metric ##
    metric = GenMetric(
        metrics=["pickscore", "aesthetic_score", "hpsv2_score", "image_reward_score"]
    )
    metric = metric.to("cuda")  # Move to GPU if available

    ## Load a dataset ##
    ## Here we assume that that images are generated and stored in a directory ##
    dataset = GenDataset(
        image_path="/data/ag/artifacts/neurips2025/partiprompts_images/llava_256/parti_long",
        caption_path="/data/ag/artifacts/neurips2025/partiprompts_images/partiprompts.txt",
    )

    dl = DataLoader(
        dataset,
        batch_size=32,  # Adjust batch size as needed
        shuffle=False,
        num_workers=4,  # Adjust number of workers based on your system
    )

    ## Evaluate the dataset ##
    for imgs, captions, _, _ in tqdm(dl):
        imgs = imgs.to("cuda")
        metric.update(
            images=imgs,
            captions=captions,
            real=False,
        )

    scores = metric.compute()
    print(scores)


if __name__ == "__main__":
    main()