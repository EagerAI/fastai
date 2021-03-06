
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
* [freeze and unfreeze](https://github.com/henry090/fastai/pull/86) a model
* object detection module - [icevision](https://github.com/henry090/fastai/issues/89)
* issue with [exporting of a pickle file](https://github.com/henry090/fastai/issues/106)

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



