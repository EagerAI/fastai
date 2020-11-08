

context("assign %f%")

source("utils.R")


test_succeeds('basic conv2d in_channel %f%', {
  conv = nn()$Conv2d(3L,3L,3L)
  init = conv[['in_channels']]
  expect_equal(conv[['in_channels']] %f% 1L, 1L)
})


test_succeeds('download mnist_sample', {
  if(!dir.exists('mnist_sample')) {
    URLs_MNIST_SAMPLE()
  }
})


test_succeeds('mnist_sample dataloader', {
  tfms = aug_transforms(do_flip = FALSE)
  path = 'mnist_sample'
  bs = 20
  data = ImageDataLoaders_from_folder(path, batch_tfms = tfms, size = 26, bs = bs)
})


test_succeeds('mnist_sample load xresnet50_deep', {
  learn = cnn_learner(data, xresnet50_deep(), metrics = accuracy)
})

test_succeeds('mnist_sample cnn xresnet50_deep channel modify', {
  init = learn$model[0][0][0][['in_channels']]
  learn$model[0][0][0][['in_channels']] %f% 1L
  expect_equal(learn$model[0][0][0][['in_channels']], init - 2)
})


test_succeeds('tensor slice', {
  abb = torch()$rand(list(3L,3L,3L))
  E(abb[0,1,1])
})










