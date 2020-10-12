## ----setup, include=FALSE-----------------------------------------------------
knitr::opts_chunk$set(echo = TRUE,eval = FALSE)

## -----------------------------------------------------------------------------
#  URLs_MNIST_SAMPLE()

## -----------------------------------------------------------------------------
#  # transformations
#  tfms = aug_transforms(do_flip = FALSE)
#  path = 'mnist_sample'
#  bs = 20
#  
#  #load into memory
#  data = ImageDataLoaders_from_folder(path, batch_tfms = tfms, size = 26, bs = bs)
#  
#  learn = cnn_learner(data, resnet18(), metrics = accuracy)

## -----------------------------------------------------------------------------
#  learn %>% fit_one_cycle(1, cbs = TerminateOnNaNCallback())

## -----------------------------------------------------------------------------
#  learn %>% fit_one_cycle(10, cbs = EarlyStoppingCallback(monitor='valid_loss', patience = 1))

## -----------------------------------------------------------------------------
#  learn = cnn_learner(data, resnet18(), metrics = accuracy, path = getwd())
#  
#  learn %>% fit_one_cycle(3, cbs = SaveModelCallback(every_epoch = TRUE,  fname = 'model'))

## -----------------------------------------------------------------------------
#  list.files('models')
#  # [1] "model_0.pth" "model_1.pth" "model_2.pth"

## -----------------------------------------------------------------------------
#  learn %>% fit_one_cycle(10, 1e-2, cbs = ReduceLROnPlateau(monitor='valid_loss', patience = 1))

## -----------------------------------------------------------------------------
#  learn %>% fit_one_cycle(10, 1e-2, cbs = ReduceLROnPlateau(monitor='valid_loss',
#  min_delta=0.1, patience = 1, min_lr = 1e-8))

## -----------------------------------------------------------------------------
#  learn = cnn_learner(data, resnet18(), metrics = accuracy, path = getwd())
#  
#  learn %>% fit_one_cycle(2, cbs = list(CSVLogger(),
#                                        ReduceLROnPlateau(monitor='valid_loss',
#                                        min_delta=0.1, patience = 1, min_lr = 1e-8)))
#  history  = read.csv('history.csv')
#  history

