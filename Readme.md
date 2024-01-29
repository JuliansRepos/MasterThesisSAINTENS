# Master Thesis: Self-Attention and Intersample Attention Transformer for Tabular Healthcare Data
Author of the master thesis: Julian Gutheil

## Setup Environments
Three different python environments are necessary:
1. autogluon_gbdt_env
   - python=3.9.11
   - autogluon=0.4.0
   - catboost=1.0.4
   - lightgbm=3.3.2
   - xgboost=1.4.2
   - scikit-learn=1.0.2
   - hyperopt=0.2.7
   - numpy=1.22.3
   - pandas=1.3.5
2. saint_env
   - python=3.9.11
   - torch=1.10.1+cu113
   - scikit-learn=1.0.2
   - einops=0.4.1
   - pandas=1.4.1
   - numpy=1.21.5
   - cudatoolkit=11.3.1
   - scipy=1.8.0
3. hopular_env
   - python=3.9.13
   - scikit-learn=1.1.1
   - scipy=1.8.1
   - pytorch=1.12.1
   - pytorch-lightning=1.7.7
   - cudatoolkit=11.3.1
   - fairscale=0.4.12
   - setuptools=59.5.0

The environments were created with the help of miniforge (https://github.com/conda-forge/miniforge).

## Directory Structure

- main_folder
  - datasets
    - downloaded_datasets
    - resources
  - bestmodels
  - Logs
  - gbdt_logs
  - models_log
  - saintens
    - models
      - model.py
      - pretrainmodel.py
    - augmentations.py
    - data.py
    - experiments.py
    - train.py
    - utils.py
    - LICENSE SAINT
  - hopular
    - auxiliary
      - data.py
    - hflayers
    - \_\_init\_\_.py
    - blocks.py
    - interactive.py
    - optim.py
    - LICENSE Hopular
  - autogluon_experiments.py
  - bayesian_hpo_tabular.py
  - dataset_prepare.py
  - Readme.md

The main_folder should always be the working directory. 
The datasets folder consists of the datasets for Autogluon,
SAINT, SAINTENS and the GBDT. The resources folder contains the datasets for Hopular. 
The downloaded_datasets folder contains the downloaded dataset files.
The folder bestmodels stores the best models of SAINT and SAINTENS.
The Logs directory stores the logs of SAINT and SAINTENS models.
The gbdt_logs directory stores the logs of the GBDT models.
The models_log directory stores the GBDT models (except LightGBM models).
The hflayers directory can be downloaded from this GitHub repo:
https://github.com/ml-jku/hopfield-layers

## Datasets
Three public and anonymized healthcare datasets from the UCI website were used:
1. diabetes dataset
   - https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008
2. thyroid dataset
   - https://archive.ics.uci.edu/dataset/102/thyroid+disease
3. myocardial dataset
   - https://archive.ics.uci.edu/dataset/579/myocardial+infarction+complications

The downloaded files should be saved in the downloaded_datasets folder.
The python script dataset_prepare.py (saint_env) was used to prepare the datasets for each method.

## Identification of Informative Features with Autogluon
The autogluon package (https://auto.gluon.ai/stable/index.html) was used in the python script autogluon_experiments.py (autogluon_gbdt_env) to identify the informative features.
The informative feature sets were implemented for SAINT and SAINTENS in the saint/data.py file, for GBDT in the BayesianHPOTabularGeneral.py file and
for Hopular in the hopular/auxiliary/data.py file.

## SAINT and SAINTENS
The SAINT and SAINTENS implementation is based on: https://github.com/somepago/saint/tree/main/old_version

The original SAINT implementation of https://github.com/somepago/saint/tree/main/old_version is under Apache 2.0 license.
The license of SAINT is in saintens/LICENSE SAINT.

The main file is saintens/train.py (saint_env).
The experiments were defined in the experiments.py file.

## Hopular
Hopular implementation comes from https://github.com/ml-jku/hopular


---------------------------------------------------
The hopular code is under MIT-style license:

Copyright (c) 2022 Institute for Machine Learning, Johannes Kepler University Linz, Austria (Bernhard Schäfl, Lukas Gruber)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
---------------------------------------------------------

The code was modified for the master thesis experiments.

The main file is hopular/interactive.py (hopular_env).

Two arguments are necessary to start the interactive.py script:
optim --dataset dataset_name

The dataset_name is not important, because the dataset will be set by the main method.

## GBDT
The Bayesian HPO implementation is oriented to this article:
https://towardsdatascience.com/an-example-of-hyperparameter-optimization-on-xgboost-lightgbm-and-catboost-using-hyperopt-12bc41a271e

Several kaggle projects used such an implementation:
https://www.kaggle.com/code/nandakishorejoshi/datefruit-catboost-with-hyperopt?source=post_page-----5d352e30778d--------------------------------
https://www.kaggle.com/code/konstantinsuloevjr/averaging-models-and-hyperopt-tuning

The Bayesian HPO for the GBDT models is implemented in the file bayesian_hpo_tabular.py (autogluon_gbdt_env).

## Disclaimer
Please note that parts of the code have their own licenses.