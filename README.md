# ChiralForce

## Dependencies
This code requires python and packages listed in the file environment.yaml. These packages are namely:
* astropy
* jupyter
* matplotlib
* numpy
* python
* scipy
* tikzplotlib
* tqdm
* texlive (UNIX) or miktex (Windows)

## Setting up the environment
* Install [Microconda](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html) and when finished open Anaconda Prompt
* Run the following command in the directory containing yaml file replacing word environment with (unix/windows): 
```
conda env create -f environment.yaml
```
* Open and run force_basis.ipynb and particles.ipynb notebooks in your favourite IDE (VSCode, JupyterLab, …)
