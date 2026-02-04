[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/release/python-31311/) ![Commit activity](https://img.shields.io/github/commit-activity/y/FrancisCrickInstitute/Spatial-Biology-Pipeline?style=plastic) ![GitHub](https://img.shields.io/github/license/FrancisCrickInstitute/Spatial-Biology-Pipeline?color=green&style=plastic)

# Overview

SopaSpan is a Python library for the analysis of spatial biology/omics data. It is heavily based on Sopa and MuSpan, combining elements of both into a single generic workflow:
* Blampey, Q., Mulder, K., Gardet, M. et al. Sopa: a technology-invariant pipeline for analyses of image-based spatial omics. _Nat Commun_ 15, 4981 (2024).
* Bull, J. A., Moore, J. W., Corry S. M., el al. MuSpAn: A Toolbox for Multiscale Spatial Analysis. _bioRxiv_ 2024.12.06.627195

<img width="350" height="350" alt="cell_type_to_cell_type" src="https://github.com/user-attachments/assets/9807689f-f471-49a5-b1ef-d701cb2db1c8" />
<img width="466" height="350" alt="umap_leiden" src="https://github.com/user-attachments/assets/ae31a25c-889b-41e6-ac7d-4af4df775a6b" />

# Installation

```bash
conda create --name spatial-bio python=3.13
conda activate spatial-bio
python -m pip install ...
```

# Usage

```bash
python sopaspan.py -i <path_to_input_file> -o <path_to_output_zarr> -p <path_to_output_plots_directory>
```
