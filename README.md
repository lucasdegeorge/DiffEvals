# DiffEvals

## How to use

Install the package from repository.

`pip install git+https://github.com/lucasdegeorge/DiffEvals.git`

```python
from genmetrics import GenMetric

## Setting up Genmetrics to calculate clip score and all of inception metrics. ... #
# ... These include FID, IS, PRDC. There is also an opportunity to use different ... #
# ... feature extractors for inception metrics calculation. ##
gen_metrics = GenMetric(
    metrics=["clip", "inception"],
    feature_extractors_for_inception_metrics=["inceptionv3", "clip", "dinov2"],
)
gen_metrics = gen_metrics.to("cuda")

## Setting the real and fake dataset ##
real_dataset = torch.rand(5000, 3, 299, 299) * 255
real_dataset = real_dataset.to(torch.uint8)

fake_dataset = torch.rand(5000, 3, 299, 299) * 255
fake_dataset = fake_dataset.to(torch.uint8)

real_loader = torch.utils.data.DataLoader(
    real_dataset, batch_size=32, shuffle=False
)
fake_loader = torch.utils.data.DataLoader(
    fake_dataset, batch_size=32, shuffle=False
)

## Looping over the real loader and fake loader separately. This is done ... #
# ... as the real dataset and fake dataset might have different number of ... #
# ... samples. In case you have same samples, please use one loop for both. ##
for real_images in tqdm(real_loader, total=len(real_loader)):
    gen_metrics.update(real_images.to("cuda"), real=True)

for fake_images in tqdm(fake_loader, total=len(fake_loader)):
    fake_text = ["abcd"] * fake_images.shape[0]
    gen_metrics.update(fake_images.to("cuda"), real=False, captions=fake_text)

eval_scores = gen_metrics.compute()

print(eval_scores)
```

