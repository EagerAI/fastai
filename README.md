## R interface to FASTAI

The fastai package provides R wrappers to [FASTAI](https://github.com/fastai/fastai).

The fastai library simplifies training fast and accurate neural nets using modern best practices. See the fastai website to get started. The library is based on research into deep learning best practices undertaken at ```fast.ai```, and includes "out of the box" support for ```vision```, ```text```, ```tabular```, and ```collab``` (collaborative filtering) models. 

[![Lifecycle: experimental](https://img.shields.io/badge/lifecycle-experimental-orange.svg)](https://www.tidyverse.org/lifecycle/#experimental)

## Installation

Requirements:

- Python >= 3.6
- CPU or GPU

The dev version:

```
devtools::install_github('henry090/fastai')
```

Later, you need to install the python module kerastuner:

```
reticulate::py_install('fastai2==0.0.30',pip = TRUE)
```

## Usage: the basics

```
library(magrittr)
library(fastai)

df = data.table::fread('https://github.com/henry090/fastai/raw/master/files/adult.csv')
```

Variables:

```
dep_var = 'salary'
cat_names = c('workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race')
cont_names = c('age', 'fnlwgt', 'education-num')
```

Preprocess strategy:

```
procs = list(FillMissing(),Categorify(),Normalize())
```

Prepare:

```
dls = TabularDataTable(df, procs, cat_names, cont_names, 
      y_names="salary", splits = list(c(1:32000),c(32001:32561))) %>% 
      dataloaders(bs=64)
```

Summary:

```
model = dls %>% tabular_learner(layers=c(200,100), metrics=accuracy)
model %>% summary()
```

```
epoch     train_loss  valid_loss  accuracy  time    
TabularModel (Input shape: ['64 x 7', '64 x 3'])
================================================================
Layer (type)         Output Shape         Param #    Trainable 
================================================================
Embedding            64 x 6               60         True      
________________________________________________________________
Embedding            64 x 8               136        True      
________________________________________________________________
Embedding            64 x 5               40         True      
________________________________________________________________
Embedding            64 x 8               136        True      
________________________________________________________________
Embedding            64 x 5               35         True      
________________________________________________________________
Embedding            64 x 4               24         True      
________________________________________________________________
Embedding            64 x 3               9          True      
________________________________________________________________
Dropout              64 x 39              0          False     
________________________________________________________________
BatchNorm1d          64 x 3               6          True      
________________________________________________________________
BatchNorm1d          64 x 42              84         True      
________________________________________________________________
Linear               64 x 200             8,400      True      
________________________________________________________________
ReLU                 64 x 200             0          False     
________________________________________________________________
BatchNorm1d          64 x 200             400        True      
________________________________________________________________
Linear               64 x 100             20,000     True      
________________________________________________________________
ReLU                 64 x 100             0          False     
________________________________________________________________
Linear               64 x 2               202        True      
________________________________________________________________

Total params: 29,532
Total trainable params: 29,532
Total non-trainable params: 0

Optimizer used: <function Adam at 0x7fa246283598>
Loss function: FlattenedLoss of CrossEntropyLoss()

Callbacks:
  - TrainEvalCallback
  - Recorder
  - ProgressCallback
```

Run:

```
model %>% fastai::fit(5,1e-2)
```

```
epoch     train_loss  valid_loss  accuracy  time    
0         0.354821    0.375355    0.825871  00:02     
1         0.366040    0.369802    0.830846  00:02     
2         0.356449    0.354734    0.830846  00:02     
3         0.356077    0.355024    0.825871  00:02     
4         0.357948    0.361930    0.835821  00:02     
```

Extract confusion matrix:

```
model %>% confusion_matrix()
```

```
     [,1] [,2]
[1,]  152    4
[2,]   29   16
```

Get predictions on new data:

```
model %>% predict(df[4,])

[1] 0.09785532 0.90214473
```


