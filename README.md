[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-31212/) ![Commit activity](https://img.shields.io/github/commit-activity/y/FrancisCrickInstitute/Spatial-Biology-Pipeline?style=plastic) ![GitHub](https://img.shields.io/github/license/FrancisCrickInstitute/Spatial-Biology-Pipeline?color=green&style=plastic)

# Overview

SopaSpan is a Python library for the analysis of spatial biology/omics data. It is heavily based on Sopa and MuSpan, combining elements of both into a single generic workflow:
* Blampey, Q., Mulder, K., Gardet, M. et al. Sopa: a technology-invariant pipeline for analyses of image-based spatial omics. _Nat Commun_ 15, 4981 (2024).
* Bull, J. A., Moore, J. W., Corry S. M., el al. MuSpAn: A Toolbox for Multiscale Spatial Analysis. _bioRxiv_ 2024.12.06.627195

<img width="350" height="350" alt="cell_type_to_cell_type" src="https://github.com/user-attachments/assets/9807689f-f471-49a5-b1ef-d701cb2db1c8" />
<img width="466" height="350" alt="umap_leiden" src="https://github.com/user-attachments/assets/ae31a25c-889b-41e6-ac7d-4af4df775a6b" />

# Installation

## Step 1: Install a Python Distribution

We recommend using conda as it's relatively straightforward and makes the management of different Python environments simple. You can install conda from [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation) (miniconda will suffice).

## Step 2: Set Up Environment

### Step 2.1: Create an environment

Once conda is installed, open a terminal (Mac) or Anaconda Prompt (Windows) and run the following series of commands:

```bash
conda create --name spatial-bio python=3.12
conda activate spatial-bio
```

You will be presented with a list of packages to be downloaded and installed. The following prompt will appear:
```bash
Proceed ([y]/n)?
```
Hit Enter and all necessary packages will be downloaded and installed - this may take some time.

### Step 2.2: Install pip dependencies

The following dependencies are required and must be installed in the correct order:

#### 2.2.1: Tensorflow

SopaSpan depends on [Stardist](https://github.com/stardist/stardist) to segment cell nuclei, which in turn depends on Tensorflow.

> [!NOTE]
> While Tensorflow will run on all operating systems, support for GPU processing is generally only supported on Linux - see [here](https://www.tensorflow.org/install) for more information.

On Linux, install tensorflow as follows:

```bash
python -m pip install tensorflow[and-cuda]
```

On any other operating system:

```bash
python -m pip install tensorflow
```

#### 2.2.2: Sopa

Install [Sopa](https://gustaveroussy.github.io/sopa/) with support for stardist and wsi (whole slide imaging) using the following:

```bash
python -m pip install 'sopa[stardist,wsi]'
```

When complete, you can deactivate the environment you have created with the following command.

```
conda deactivate
```
You have successfully set up an environment to run KimmelNet!


# Usage

```bash
python sopaspan.py -i <path_to_input_file> -o <path_to_output_zarr> -p <path_to_output_plots_directory>
```
