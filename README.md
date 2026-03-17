# DiffEvals

## TODO
- [x] Implement HPSv2, PickScore, Aesthetic Score.
- [x] Implement Image-reward.
- [x] Fix the installation procedure.
- [ ] Implement GenEval.
- [ ] Cleanup with separate files for aesthetic scores.

## Installation

Install the package from repository.

`pip install git+https://github.com/lucasdegeorge/DiffEvals.git`

The dependencies will be automatically installed.

One other alternative (and the recommended one) is to use `uv` for installation. Its super fast. For doing that you can already use `uv add git+https://github.com/lucasdegeorge/DiffEvals.git`. You could also just `git clone https://github.com/lucasdegeorge/DiffEvals.git` and then do `uv sync` to have everything installed super fast.

## Usage

### Simple Usage

```python
from genmetrics import GenMetric

## Setting up Genmetrics to calculate clip score, clip score with jina backbone ... #
# and all of inception metrics. These include FID, IS, PRDC. There is also an ... #
# opportunity to use different feature extractors for inception metrics calculation. ##
gen_metrics = GenMetric(
    ## For inception metrics add inception_ prefix ##
    metrics=["inception_fid", "inception_is" "clip", "jina_clip"], 
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
# ... samples. In case you have same number of samples, please use one loop for both. ##
for real_images in tqdm(real_loader, total=len(real_loader)):
    gen_metrics.update(images=real_images.to("cuda"), real=True)

for fake_images in tqdm(fake_loader, total=len(fake_loader)):
    captions = ["abcd"] * fake_images.shape[0]
    gen_metrics.update(fake_images.to("cuda"), real=False, captions=captions)

eval_scores = gen_metrics.compute()

print(eval_scores)
```

### `GenDataset` based usage

#### 1. Usage where you have already prepared the datasets

One could also use the in-built `GenDataset` to create the dataset and calculate scores. We can also have `GenDataset` with `.npz` file with no images inside them and only mean and std stats. 


```python
from genmetrics import GenMetric, GenDataset

## Setting up Genmetrics to calculate clip score and all of inception metrics. ... #
# ... These include FID, IS, PRDC. There is also an opportunity to use different ... #
# ... feature extractors for inception metrics calculation. ##
gen_metrics = GenMetric(
    metrics=["inception_fid"],
    feature_extractors_for_inception_metrics=["inceptionv3", "clip", "dinov2"],
)
gen_metrics = gen_metrics.to("cuda")

## Setting the real and fake dataset ##
## 
real_dataset = GenDataset(
    image_path="real/img/folder/or/npz", 
    ## This suggests no images are there in the npz file! (only for inception_fid)
    return_no_images=True, 
    dataset_size=50000
) # the image_path can also be a folder!

fake_dataset = GenDataset(
    image_path="fake/img/folder/or/npz", 
    ref_caption_path="/ref/caption/path/txt"
) # the image_path can also be a folder!

real_loader = torch.utils.data.DataLoader(
    real_dataset, batch_size=32, shuffle=False
)
fake_loader = torch.utils.data.DataLoader(
    fake_dataset, batch_size=32, shuffle=False
)

## Looping over the real loader and fake loader separately. This is done ... #
# ... as the real dataset and fake dataset might have different number of ... #
# ... samples. In case you have same number of samples, please use one loop for both. ##
for batch in tqdm(real_loader, total=len(real_loader)):
    gen_metrics.update(
        images=batch.get("image", None),
        mean=batch["mean"], 
        sigma=batch["sigma"],
        real=True
    )

for batch in tqdm(fake_loader, total=len(fake_loader)):
    gen_metrics.update(
        images=batch.get("image", None), # Typically shouldn't be None! 
        mean=batch["mean"], 
        sigma=batch["sigma"],
        real=False, 
        captions=batch["ref_captions"]
    )

eval_scores = gen_metrics.compute()

print(eval_scores)

```

#### 2. Usage where you have already prepared real dataset but generate the fake dataset.
```python
## ... same code as before ... ##
fake_dataset = GenDataset(
    image_path=None, 
    ref_caption_path="/ref/caption/path/txt",
    gen_caption_path="/gen/caption/path/txt",
) # the image_path can also be a folder!

## ... code from before ... ##
for batch in tqdm(fake_loader):
    generated_images = GenerationModel(caption=batch["gen_caption"], ...)
    gen_metrics.update(
        images=generated_images,
        mean=batch["mean"], 
        sigma=batch["sigma"],
        real=False, 
        captions=batch["ref_captions"]
    )

```

One could also just use the `InceptionMetric` and `CLIPJinaScore` separately by doing the following.
```python
from genmetrics import InceptionMetric, CLIPJinaScore
```
Check the classes in `src.genmetrics.evaluations` for more information.

Additionally we also have the opportunity to calculate aesthetics metrics like `PickScore`, `AestheticScore`, `HPSv2` and `ImageReward`. The use case is quite similar as before but to calculate these you dont need to have a real dataset and hence the code can be readily simplified by just using one dataloader.
