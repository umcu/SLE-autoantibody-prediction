# SLE

This project aims to improve the diagnosis of [Systemic Lupus Erythematosus](https://en.wikipedia.org/wiki/Systemic_lupus_erythematosus) using text mining and predictive modelling. The code in this repo belongs to the following publication:

Brunekreef TE, Reteig LC, Limper M, Haitjema S, Dias J, Mathsson-Alm L, van Laar JM, Otten HG. [Microarray analysis of autoantibodies can identify future Systemic Lupus Erythematosus patients. Human Immunology](https://doi.org/10.1016/j.humimm.2022.03.010). 2022 Jun 1;83(6):509-14.

The `notebooks/Main Results.ipynb` notebook contains all the results that were published in this paper. The other notebooks in this folder contain all the other (exploratory) analyses that we ran for which we were able to publish the code.

See the other READMEs in the folders for more information.

## Text mining

Clinical data (whether a person is considered to have SLE or not) is extracted from the electronic health records. 

See the (R) scripts in `scripts/text-mining` and the `text-mining-validation` notebook in `notebooks`.

## Predictive modelling

The next step is to use this data to predict SLE diagnosis using blood biomarkers such as auto-antibodies. 

See the (Python) code in `src/sle` and the `notebooks` folder.

### Reproducibility

N.B. The original patient data was not shared along with the publication. If you don't have access to this data, but would still like to run the code, the notebooks offer the option to generate some simple synthetic data.

1. [Install a version of the conda management system](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) (e.g. Miniconda), if you don't have it already.
2. Clone this repo, e.g.: 
    ```zsh
    git clone https://github.com/umcu/negation-detection.git
    cd negation-detection
    ```
3. Make and activate the virtual environment, e.g.:
    ```zsh
    conda env create -f environment.yml
    conda activate SLE
    ```
4. Install the project package (see `src/sle`)
    ```zsh
    pip install -e .
    ```
5. Open and run any of the notebooks, for instance with JupyterLab:
    ```zsh
    jupyter lab
    ```



