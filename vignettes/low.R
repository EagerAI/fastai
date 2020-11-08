## ----setup, include=FALSE-----------------------------------------------------
knitr::opts_chunk$set(echo = TRUE,eval = FALSE,echo = T)

## -----------------------------------------------------------------------------
#  URLs_MNIST_SAMPLE()
#  tfms = aug_transforms(do_flip = FALSE)
#  path = 'mnist_sample'
#  bs = 20
#  data = ImageDataLoaders_from_folder(path, batch_tfms = tfms, size = 26, bs = bs)
#  learn = cnn_learner(data, xresnet50_deep(), metrics = accuracy)

## -----------------------------------------------------------------------------
#  init = learn$model[0][0][0][['in_channels']]
#  print(init)
#  # 3
#  learn$model[0][0][0][['in_channels']] %f% 1L
#  print(learn$model[0][0][0][['in_channels']])
#  # 1

## -----------------------------------------------------------------------------
#  names(learn$model[0][0][0])

## -----------------------------------------------------------------------------
#  print(learn$model[0][0][0])
#  # Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False))
#  learn$model[0][0][0][['kernel_size']] %f%  reticulate::tuple(list(9L,9L))
#  # Conv2d(1, 32, kernel_size=(9, 9), stride=(2, 2), padding=(1, 1), bias=False)

## -----------------------------------------------------------------------------
#  x = tensor(c(1,2), c(3,4))
#  print(x[0][0])
#  # tensor(1.)
#  
#  # Now change it to 99.
#  x[0][0] %f% 99
#  print(x[0][0])
#  # tensor(99.)

## -----------------------------------------------------------------------------
#  print(x[0])
#  # tensor([99.,  2.])
#  # change to 55, 55
#  x[0] %f% c(55,55)
#  # tensor([55., 55.])

