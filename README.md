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
reticulate::py_install('fastai',pip = TRUE)
```

## Usage: the basics

```
library(fastai)
library(magrittr)

df = data.table::fread('~/Downloads/adult_sample/adult.csv')
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

Split:

```
test = tabular_TabularList_from_df(df[800:1000,], 
                                   cat_names=cat_names, cont_names=cont_names)
data = tabular_TabularList_from_df(df, cat_names=cat_names, 
                                   cont_names=cont_names, procs=procs)
```

Prepare:

```
baked <- data %>% split_by_idx(800:1000) %>% 
  label_from_df(dep_var) %>% 
  add_test(test) %>% 
  databunch()
```

Summary:

```
model = baked %>% tabular_learner(layers=list(200L,100L), metrics=accuracy,
                                path=getwd())

summary(model)
```

```
TabularModel
======================================================================
Layer (type)         Output Shape         Param #    Trainable 
======================================================================
Embedding            [6]                  60         True      
______________________________________________________________________
Embedding            [8]                  136        True      
______________________________________________________________________
Embedding            [5]                  40         True      
______________________________________________________________________
Embedding            [8]                  136        True      
______________________________________________________________________
Embedding            [5]                  35         True      
______________________________________________________________________
Embedding            [4]                  24         True      
______________________________________________________________________
Embedding            [3]                  9          True      
______________________________________________________________________
Dropout              [39]                 0          False     
______________________________________________________________________
BatchNorm1d          [3]                  6          True      
______________________________________________________________________
Linear               [200]                8,600      True      
______________________________________________________________________
ReLU                 [200]                0          False     
______________________________________________________________________
BatchNorm1d          [200]                400        True      
______________________________________________________________________
Linear               [100]                20,100     True      
______________________________________________________________________
ReLU                 [100]                0          False     
______________________________________________________________________
BatchNorm1d          [100]                200        True      
______________________________________________________________________
Linear               [2]                  202        True      
______________________________________________________________________

Total params: 29,948
Total trainable params: 29,948
Total non-trainable params: 0
Optimized with 'torch.optim.adam.Adam', betas=(0.9, 0.99)
Using true weight decay as discussed in https://www.fast.ai/2018/07/02/adam-weight-decay/ 
Loss function : FlattenedLoss
======================================================================
Callbacks functions applied 
```

Run:

```
model %>% fit(10, 1e-2)
```

```
epoch     train_loss  valid_loss  accuracy  time    
0         0.354821    0.375355    0.825871  00:02     
1         0.366040    0.369802    0.830846  00:02     
2         0.356449    0.354734    0.830846  00:02     
3         0.356077    0.355024    0.825871  00:02     
4         0.357948    0.361930    0.835821  00:02     
5         0.347525    0.352505    0.860696  00:02     
6         0.349367    0.341253    0.860696  00:02     
7         0.351288    0.337538    0.840796  00:02     
8         0.358390    0.343998    0.845771  00:02     
9         0.352725    0.337811    0.850746  00:02 
```




