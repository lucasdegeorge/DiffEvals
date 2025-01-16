from setuptools import setup, find_packages

setup(
    name="genmetrics",
    version="0.1",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "numpy",
        "torch",
        "torchvision",
        "torchaudio",
        "torchmetrics",
        "tqdm",
        "transformers",
        "torch-fidelity",
        "scipy",
        "prdc",
        "timm",
    ],
    url="https://github.com/lucasdegeorge/DiffEvals",
    author="Lucas Degeorge, Arijit Ghosh",
    license="MIT",
)