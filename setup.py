import os
import sys
from setuptools import setup, find_packages

install_requires = [
    "numpy>=1.19.5",
    "pandas>=1.2.1",
    "scipy >=1.4.1",
    "anndata>=0.7.5"
    "scikit-learn>=0.24.2",
    "matplotlib",
    "seaborn>=0.11.1",
    "statsmodels>=0.12.2"
]

if sys.version_info[:2] < (3, 7):
    raise RuntimeError("Python version >=3.7 required.")

with open("README.md", "r", encoding="utf-8") as fh:
    readme = fh.read()

setup(
    name="spatialcorr",
    version="1.2.0",
    description="SpatialCorr",
    author="Matthew N. Bernstein",
    author_email="mbernstein@morgridge.org",
    packages=[
        "spatialcorr"
    ],
    license="MIT License",
    install_requires=install_requires,
    long_description=readme,
    package_data={'spatialcorr':['datasets/GSM4284326_P10_ST_rep2.h5ad']},
    include_package_data=True,
    zip_safe=True,
    url="https://github.com/mbernste/spatialcorr",
    entry_points={},
    keywords=[
        "spatial-transcriptomics",
        "gene-expression",
        "computational-biology"
    ],
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX :: Linux",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ]
)


