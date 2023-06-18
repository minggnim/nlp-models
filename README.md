<a href="https://github.com/minggnim/nlp-models/blob/master/LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-success" alt="Apache 2.0 License"></a>
[![Python package](https://github.com/minggnim/nlp-classification-model/actions/workflows/python-package.yml/badge.svg)](https://github.com/minggnim/nlp-classification-model/actions/workflows/python-package.yml)
[![Upload Python Package](https://github.com/minggnim/nlp-classification-model/actions/workflows/python-publish.yml/badge.svg)](https://github.com/minggnim/nlp-classification-model/actions/workflows/python-publish.yml)
[![Dependency Review](https://github.com/minggnim/nlp-classification-model/actions/workflows/dependency-review.yml/badge.svg)](https://github.com/minggnim/nlp-classification-model/actions/workflows/dependency-review.yml)

# NLP Models

A repository for building transformer based nlp models

## Models

1. bert_classifier
   A wrapper package around BERT-based classification models
   
   - [Training example](https://github.com/minggnim/nlp-models/blob/master/notebooks/01_a_classification_model_training_example.ipynb)
   - [Inference example](https://github.com/minggnim/nlp-models/blob/master/notebooks/01_b_classification_inference_example.ipynb)
   
3. multi_task_model
   An implementation of multi-tasking model built on encoder models
   
   - [Zero-shot multi-task model](https://github.com/minggnim/nlp-models/blob/master/notebooks/02_a_multitask_model_zeroshot_learning.ipynb)
   - [Training example](https://github.com/minggnim/nlp-models/blob/master/notebooks/02_b_multitask_model_training_example.ipynb)
   - [Inference example](https://github.com/minggnim/nlp-models/blob/master/notebooks/02_c_multitask_model_inference_example.ipynb)

## Other Example Notebooks

- [Training GPT-2 model](https://github.com/minggnim/nlp-models/blob/master/notebooks/03_gpt2_training.ipynb)
- [Running Falcon 4b model](https://github.com/minggnim/nlp-models/blob/master/notebooks/04_falcon_4b.ipynb)

## Installation

### Install from PyPi

```
pip install nlp-models
```

### Install from source

```
git clone git@github.com:minggnim/nlp-models.git
pip install -r requirements
```
