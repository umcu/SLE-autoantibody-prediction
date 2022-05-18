# SLE

This project aims to improve the diagnosis of [Systemic Lupus Erythematosus](https://en.wikipedia.org/wiki/Systemic_lupus_erythematosus) using predictive modelling. The code in this repo belongs to the following publication:

Brunekreef TE, Reteig LC, Limper M, Haitjema S, Dias J, Mathsson-Alm L, van Laar JM, Otten HG. [Microarray analysis of autoantibodies can identify future Systemic Lupus Erythematosus patients. Human Immunology](https://doi.org/10.1016/j.humimm.2022.03.010). 2022 Jun 1;83(6):509-14.

## Contents of this repository

- `data/` contains just a README describing the data files; the original patient data were not shared along with the publication. If you don't have access to this data, but would still like to run the code, the notebooks offer the option to generate some simple synthetic data
- `notebooks/` contains several jupyter notebooks that contain all the (exploratory) analyses that we ran (for which we were able to publish the code).
    - `notebooks/Main Results.ipynb` notebook contains all the results that were published in the paper
- `src/sle/` contains a number of python modules with supporting code that are imported in the notebooks

## Reproducibility

1. You'll need either the `conda` or the `mamba` package manager to recreate the computational environment. It might take `conda` a while to resolve the environment in `environment.yml` (see step 3), so it's recommended to use mamba instead.

    If you already have `conda` installed, you can install mamba as follows:

    ```zsh
    conda install mamba -n base -c conda-forge
    ```

    If you don't have conda, you can skip it and [install `mambaforge`](https://github.com/conda-forge/miniforge#mambaforge) instead:

    ```zsh
    wget "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
    bash Mambaforge-$(uname)-$(uname -m).sh
    ```
2. Clone this repo, e.g.: 
    ```zsh
    git clone https://github.com/umcu/SLE-autoantibody-prediction
    cd SLE-autoantibody-prediction
    ```
3. Make and activate the virtual environment, e.g.:
    ```zsh
    mamba env create -f environment.yaml
    conda activate SLE
    ```
4. Install the project package (see `src/sle`), e.g.
    ```zsh
    pip install -e .
    ```
5. Open and run any of the notebooks, for instance with JupyterLab:
    ```zsh
    jupyter lab
    ```



