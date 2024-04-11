
## 2.2.2

- Updated installation function ```install_fastai```

## 2.2.1

- Fixed breaking changes during model training
- Updated installation function ```install_fastai```

## 2.0.9

- PyTorch version is 1.9
- lr_find bug is fixed


## 2.0.8

- minor fixes

## 2.0.7

- bs_finder is fixed
- better visualization on Colab. Issue with fig size is fixed

## 2.0.6

- new function to [load_learner](https://github.com/EagerAI/fastai/issues/115)
- unet_config is [Deprecated](https://github.com/EagerAI/fastai/issues/128)
- while installing fast.ai Mac OS, first, it downloads PyTorch 1.8, then 1.7.1. It is fixed, [now](https://github.com/EagerAI/fastai/issues/129).
- ```nn_module()``` function allows to rename the model, e.g. ```summary(model)```
- ```nn_module()``` will not move the model to GPU, if ```gpu``` argument is *FALSE* (by default it is *TRUE*)
- [custom loss functions](https://github.com/EagerAI/fastai/pull/132) with ```nn_loss()```. Based on [Kaggle notebook](https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch)

## 2.0.5 

- ```install_fastai``` no more supports extensions. They need to be installed separately by users.
- PyTorch was upgraded from 1.7.0 to 1.7.1.

## 2.0.4 

* stick to fastaudio 0.1.3 (resolve dependencies)
* add ```geom_point``` for interactive visualization within RStudio
* [add TPU module into fastai](https://colab.research.google.com/drive/1PiBECDM552No-5apVIB8LqUSdSqqJSi-?usp=sharing)

## 2.0.3 

* current stable version of ```fast.ai``` is ```2.1.5```
* lots of new callback ops
* [freeze and unfreeze](https://github.com/EagerAI/fastai/pull/86) a model
* object detection module - [icevision](https://github.com/EagerAI/fastai/issues/89)
* issue with [exporting of a pickle file](https://github.com/EagerAI/fastai/issues/106)

## 2.0.2 

* Hugging Face integration, prediction
* ```one_batch()``` ability to add more arguments
* no need to call ```options(reticulate.useImportHook = FALSE)```
* ```DataBlock``` automatically places data into ```cuda``` if available


## 2.0.1

* ```nn_module``` for model construction
* ```fix_fit``` for disabling the training plot
* all the ```fit``` functions now return the training history


## fastai 2.0.0

* Initial release



