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
#  # Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False))
#  learn$model[0][0][0][['kernel_size']] %f%  reticulate::tuple(list(9L,9L))
#  print(learn$model[0][0][0])
#  # Conv2d(1, 32, kernel_size=(9, 9), stride=(2, 2), padding=(1, 1), bias=False)

## -----------------------------------------------------------------------------
#  x = tensor(c(1,2), c(3,4))
#  # tensor([[1., 2.],
#  #         [3., 4.]])
#  print(x[0][0])
#  # tensor(1.)
#  
#  # Now change it to 99.
#  x[0][0] %f% 99
#  print(x[0][0])
#  # tensor(99.)
#  
#  print(x)
#  # tensor([[99.,  2.],
#  #         [ 3.,  4.]])

## -----------------------------------------------------------------------------
#  print(x[0])
#  # tensor([99.,  2.])
#  # change to 55, 55
#  x[0] %f% c(55,55)
#  # tensor([55., 55.])

## -----------------------------------------------------------------------------
#  a = tensor(array(1:100, c(3,3,3,3)))
#  a$shape
#  # torch.Size([3, 3, 3, 3])

## -----------------------------------------------------------------------------
#  a %>% narrow('[0,:,:,:]')

## -----------------------------------------------------------------------------
#  a %>% narrow("[:,0,:,:]")

## -----------------------------------------------------------------------------
#  a %>% narrow('[:,0,0,:]')

## -----------------------------------------------------------------------------
#  a %>% narrow("[1,1,1,:]")

## -----------------------------------------------------------------------------
#  library(magrittr)
#  library(fastai)
#  library(zeallot)
#  
#  if(!file.exists('mnist.pkl.gz')) {
#    download.file('http://deeplearning.net/data/mnist/mnist.pkl.gz','mnist.pkl.gz')
#    R.utils::gunzip("mnist.pkl.gz", remove=FALSE)
#  }
#  
#  c(c(x_train, y_train), c(x_valid, y_valid), res) %<-%
#    reticulate::py_load_object('mnist.pkl', encoding = 'latin-1')
#  
#  x_train = x_train[1:500,1:784]
#  x_valid = x_valid[1:500,1:784]
#  
#  y_train = as.integer(y_train)[1:500]
#  y_valid = as.integer(y_valid)[1:500]
#  

## -----------------------------------------------------------------------------
#  example = array_reshape(x_train[1,], c(28,28))
#  
#  example %>% show_image(cmap = 'gray') %>% plot()

## -----------------------------------------------------------------------------
#  TensorDataset = torch()$utils$data$TensorDataset
#  
#  bs = 32
#  train_ds = TensorDataset(tensor(x_train), tensor(y_train))
#  valid_ds = TensorDataset(tensor(x_valid), tensor(y_valid))
#  train_dl = TfmdDL(train_ds, bs = bs, shuffle = TRUE)
#  valid_dl = TfmdDL(valid_ds, bs = 2 * bs)
#  dls = Data_Loaders(train_dl, valid_dl)
#  
#  one = one_batch(dls)
#  x = one[[1]]
#  y = one[[2]]
#  x$shape; y$shape
#  
#  nn = nn()
#  Functional = torch()$nn$functional

## -----------------------------------------------------------------------------
#  model = nn_module(function(self) {
#  
#    self$lin1 = nn$Linear(784L, 50L, bias=TRUE)
#    self$lin2 = nn$Linear(50L, 10L, bias=TRUE)
#  
#    forward = function(y) {
#      x = self$lin1(y)
#      x = Functional$relu(x)
#      self$lin2(x)
#    }
#  })

## -----------------------------------------------------------------------------
#  learn = Learner(dls, model, loss_func=nn$CrossEntropyLoss(), metrics=accuracy)
#  
#  learn %>% summary()
#  
#  learn %>% fit_one_cycle(1, 1e-2)

