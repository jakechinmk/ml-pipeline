# Machine Learning Pipeline

## Installation and Settings

### Base Requirements to run this repo

- git clone this repository, and navigate to it:

```bash
git clone jakechinmk/data_science
cd ml-pipeline
```

- Install [poetry](https://python-poetry.org/docs/#installing-with-the-official-installer) for dependency management
- Install ``make`` for scripts
  - osx: (Using homebrew): ``brew install make``
  - windows: (Using chocalatey) ``choco install make``
- Ensure that you have a python version >= 3.8, < 3.12 version installed in your machine.

### Install dependencies

```bash
poetry install
```

Verify everything is working by running ``make env``

PS: There's are also better way to config the environment using docker images. This will be in the next enhancement instead.

## Guides

### Basic Commands

- Install environment using poetry: ``make env``
- Preprocessing: ``make preprocess``
- Exploration: ``make exploration``
- Train: ``make train``
- Inference: ``make inference``
- Show experiments run with user interface: ``make validator``
- Model Explainer: ``make validate``

### Configurations Details

Current repository are heavily dependent on ``config.yaml`` which sit in ``config`` folder.

Config.yaml file will have a few sections:

- pycaret
  - setup (parameters to be used in pycaret [setup](https://pycaret.readthedocs.io/en/latest/api/classification.html#pycaret.classification.setup) function)
  - model (parameters to control model to be used)
- preprocessing (current unusable)
- overall (mainly for defining path)
  - if the preprocess pipeline is not used, the current inference pipeline might not be functional too. Will require some testing

### Strategy

Based on the ``exploration_raw.html``, the missing value in each feature is not more than 50%, hence it's okay to impute values. Median is choosen in general as it is more robust to outlier. Several features are bin to avoid outliers as well. The target is imbalance, hence the current handling method will be using SMOTE to generate synthetic data sample (this will need to be handle with care, will need to check on the test dataset performance instead of test dataset performance). The current metric will be AUC although it will be better using recall (cost of losing loan capital > cost of losing a potential customer). Based on the model performance, it will be expected to spend more effort to understand what will be the error lies in instead of just checking a few metrics. 


## Objectives

To create a basic end-to-end machine learning pipeline for classification (regression support is not included.)

It will be best to use low-codes tool which provide good enough functionality and build rapid prototypes due to time constraint. [Pycaret](https://pycaret.org/) is introduce here which help to

- Track experiment
- Build ML application
- Create REST API
- Build docker image

The current structure is not as per suggested in the documents. That will be consider part of the enhancment in the future if the code based expand.

## Backlog

- The current design is not robust enough to handle every situation
  - prepocessing steps: cannot take in extra self-defined cleaning pipeline.
  - training steps: have not included the list to exclude when doing comparison
  - validator steps: the pipeline is taking too slow while running explainer dashboard
  - validate steps: mlflow is not recording the plots in a proper manner
  - multiple class are using the same function, will need to redesign such that they inherit from a main base class.
- Model performance evaluation
  - Could be better by including samples that is wrongly predicted in the holdout sample
  - Generating the plots are taking too long time
- Test script
