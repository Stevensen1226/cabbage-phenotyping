from setuptools import setup, find_packages

setup(
    name="cabbage_pheno",
    version="0.1.0",
    description="Automatic extraction of cabbage phenotype parameters from point clouds",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "open3d",
        "scikit-learn",
        "pyyaml",
        "tqdm",
        "matplotlib"
    ],
)
