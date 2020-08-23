## R interface to FASTAI

The fastai package provides R wrappers to [FASTAI](https://github.com/fastai/fastai).

The fastai library simplifies training fast and accurate neural nets using modern best practices. See the fastai website to get started. The library is based on research into deep learning best practices undertaken at ```fast.ai```, and includes "out of the box" support for ```vision```, ```text```, ```tabular```, and ```collab``` (collaborative filtering) models. 

[![Lifecycle: experimental](https://img.shields.io/badge/lifecycle-experimental-orange.svg)](https://www.tidyverse.org/lifecycle/#experimental)
![fastai_stable](https://github.com/henry090/fastai/workflows/fastai2_stable/badge.svg)
![fastai2_win](https://github.com/henry090/fastai/workflows/fastai2_win/badge.svg)

## Installation

Requirements:

- Python >= 3.6
- CPU or GPU

The dev version:

```
devtools::install_github('henry090/fastai')
```

Later, you need to install the python module fastai:

```
reticulate::py_install('fastai2==0.0.30',pip = TRUE)
```

## Tabular data

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
model %>% get_confusion_matrix()
```

```
       <50k  >=50k
<50k   407    22
>=50k   68    64
```

Get predictions on new data:

```
model %>% predict(df[4,])

[1] 0.09785532 0.90214473
```

## Image data

Get Pets dataset:

```
fastai::URLs_PETS()
```

Define path to folders:

```
path = 'oxford-iiit-pet'
path_anno = 'oxford-iiit-pet/annotations'
path_img = 'oxford-iiit-pet/images'
fnames = get_image_files(path_img)
```

See one of examples:

```
fnames[1]

oxford-iiit-pet/images/american_pit_bull_terrier_129.jpg
```

Load the data into GPU/CPU:

```
dls = ImageDataLoaders_from_name_re(
  path, fnames, pat='(.+)_\\d+.jpg$',
  item_tfms=Resize(size = 460), bs = 10,
  batch_tfms=list(aug_transforms(size = 224, min_scale = 0.75),
                  Normalize_from_stats( imagenet_stats() )
                  ),
  device = 'cuda'
)
```

Random batch for visualization:

```
par(mar=c(0.5, 0.5, 1, 1))

imager::map_il(dls %>% fastai::random_batch(),
               imager::load.image) %>% plot(axes=FALSE)
```

<img src="files/pets.png" geight=500 align=center alt="Pets"/>

Model architecture:

```
learn = cnn_learner(dls, resnet34, metrics = error_rate)
```

And fit:

```
learn %>% fit_one_cycle(n_epoch = 2)

epoch     train_loss  valid_loss  error_rate  time
0         0.904872    0.317927    0.105548    00:35
1         0.694395    0.239520    0.083897    00:36
```

Get confusion matrix and plot:

```
conf = learn %>% fastai::get_confusion_matrix()

library(highcharter)
hchart(conf,label=TRUE) %>%
    hc_yAxis(title = list(text='Actual')) %>%
    hc_xAxis(title = list(text='Predicted'),
             labels = list(rotation=-90))
```

<img src="files/conf.png" geight=500 align=center alt="Pets"/>

> Note that the plot is built with highcharter.

Alternatively, load images from folders:

```
# get sample data
URLs_MNIST_SAMPLE()

# transformations
tfms = aug_transforms(do_flip=FALSE)
path='mnist_sample'
bs=20

#load into memory
data = ImageDataLoaders_from_folder(path, batch_tfms = tfms, size = 26, bs=bs)

# Visualize and train 
par(mar=c(0.5, 0.5, 1, 1))

imager::map_il(data %>% fastai::random_batch(regex = '[0-9]+'),
               imager::load.image) %>% plot(axes=FALSE)
               
learn = cnn_learner(data, resnet18, metrics=accuracy)
learn %>% fit(2)
```

<img src="files/mnist.png" geight=500 align=center alt="Pets"/>







