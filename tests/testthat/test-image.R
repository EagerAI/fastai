context("image")

source("utils.R")

test_succeeds('download mnist_sample', {
  if(!dir.exists('mnist_sample')) {
    URLs_MNIST_SAMPLE()
  }
})


test_succeeds('mnist_sample transformations', {
  tfms = aug_transforms(do_flip = FALSE)
  path = 'mnist_sample'
  bs = 20
  expect_length(tfms, 2)
})

test_succeeds('mnist_sample load into memory', {
  data = ImageDataLoaders_from_folder(path, batch_tfms = tfms, size = 26, bs = bs)
  expect_length(one_batch(data, convert = FALSE),2)
  expect_length(one_batch(data, TRUE),2)
  expect_length(one_batch(data,TRUE)[[2]], data$bs)
  expect_length(one_batch(data,TRUE)[[1]], data$bs)
  expect_equal(dim(one_batch(data,TRUE)[[1]][[1]]), c(28, 28, 3))
})

test_succeeds('mnist_sample cnn_learner', {
  learn = cnn_learner(data, resnet18, metrics = accuracy)
})

#test_succeeds('mnist_sample predict', {
#  result = learn %>% predict(list.files('mnist_sample',recursive = TRUE, full.names = TRUE)[10])
#  expect_length(result, 2)
#  expect_equal(result[[2]], "3")
#  expect_equal(names(result[[1]]), c('3','7'))
#})






