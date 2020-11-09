

context("nn_module")

source("utils.R")


test_succeeds('download MNIST pickler', {
  download.file('http://deeplearning.net/data/mnist/mnist.pkl.gz','mnist.pkl.gz')
  R.utils::gunzip("mnist.pkl.gz", remove=FALSE)
})

test_succeeds('fit nn_module MNIST', {
  object = reticulate::py_load_object('mnist.pkl', encoding='latin-1')

  x_train = object[[1]][[1]][1:500,1:784]
  x_valid = object[[2]][[1]][1:500,1:784]

  y_train = as.integer(object[[1]][[2]])[1:500]
  y_valid = as.integer(object[[2]][[2]])[1:500]

  example = array_reshape(x_train[1,], c(28,28))

  example %>% show_image(cmap = 'gray') %>% plot()

  TensorDataset = torch()$utils$data$TensorDataset

  bs = 32
  train_ds = TensorDataset(tensor(x_train), tensor(y_train))
  valid_ds = TensorDataset(tensor(x_valid), tensor(y_valid))
  train_dl = TfmdDL(train_ds, bs = bs, shuffle = TRUE)
  valid_dl = TfmdDL(valid_ds, bs = 2 * bs)
  dls = Data_Loaders(train_dl, valid_dl)


  c(x,y) %<-% one_batch(dls)
  x$shape; y$shape

  nn = nn()
  Functional = torch()$nn$functional

  my_module = function(self) {

    self$lin1 = nn$Linear(784L, 50L, bias=TRUE)
    self$lin2 = nn$Linear(50L, 10L, bias=TRUE)

    forward = function(y) {
      x = self$lin1(y)
      x = Functional$relu(x)
      self$lin2(x)
    }
  }

  model = nn_module(my_module)

  learn = Learner(dls, model, loss_func=nn$CrossEntropyLoss(), metrics=accuracy)

  learn %>% summary()

  learn %>% fit_one_cycle(1, 1e-2)
})


