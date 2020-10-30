
context("pretrained_model_weights with MNIST")

source("utils.R")


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
  summary(learn)
})

test_succeeds('mnist_sample load xresnet50', {
  learn = cnn_learner(data, xresnet50(), metrics = accuracy)
  summary(learn)
})


test_succeeds('mnist_sample load xresnet34_deep', {
  learn = cnn_learner(data, xresnet34_deep(), metrics = accuracy)
  summary(learn)
})

test_succeeds('mnist_sample load xresnet34', {
  learn = cnn_learner(data, xresnet34(), metrics = accuracy)
  summary(learn)
})

test_succeeds('mnist_sample load xresnet18_deep', {
  learn = cnn_learner(data, xresnet18_deep(), metrics = accuracy)
  summary(learn)
})


test_succeeds('mnist_sample load xresnet18', {
  learn = cnn_learner(data, xresnet18(), metrics = accuracy)
  summary(learn)
})


test_succeeds('mnist_sample load xresnet152', {
  learn = cnn_learner(data, xresnet152(), metrics = accuracy)
  summary(learn)
})

test_succeeds('mnist_sample load xresnet101', {
  learn = cnn_learner(data, xresnet101(), metrics = accuracy)
  summary(learn)
})


test_succeeds('mnist_sample load xresnet50_deep', {
  learn = cnn_learner(data, xresnet50_deep(), metrics = accuracy)
  summary(learn)
})

test_succeeds('download PETS', {
  if(!dir.exists('oxford-iiit-pet')) {
    URLs_PETS()
  }
  path = 'oxford-iiit-pet'
  path_anno = 'oxford-iiit-pet/annotations'
  path_img = 'oxford-iiit-pet/images'
  fnames = get_image_files(path_img)

  dls = ImageDataLoaders_from_name_re(
    path, fnames, pat='(.+)_\\d+.jpg$',
    item_tfms=Resize(size = 460), bs = 10,
    batch_tfms=list(aug_transforms(size = 224, min_scale = 0.75),
                    Normalize_from_stats( imagenet_stats() )
    )
  )
})

test_succeeds('pet load alexnet', {
  learn = cnn_learner(dls, alexnet(), metrics = accuracy)
  summary(learn)
})

test_succeeds('pet load vgg19_bn', {
  learn = cnn_learner(dls, vgg19_bn(), metrics = accuracy)
  summary(learn)
})

test_succeeds('pet load vgg16_bn', {
  learn = cnn_learner(dls, vgg16_bn(), metrics = accuracy)
  summary(learn)
})

test_succeeds('pet load vgg13_bn', {
  learn = cnn_learner(dls, vgg13_bn(), metrics = accuracy)
  summary(learn)
})


test_succeeds('pet load vgg11_bn', {
  learn = cnn_learner(dls, vgg11_bn(), metrics = accuracy)
  summary(learn)
})

test_succeeds('pet load squeezenet1_1', {
  learn = cnn_learner(dls, squeezenet1_1(), metrics = accuracy)
  summary(learn)
})


test_succeeds('pet load squeezenet1_0', {
  learn = cnn_learner(dls, squeezenet1_0(), metrics = accuracy)
  summary(learn)
})

test_succeeds('pet load resnet50', {
  learn = cnn_learner(dls, resnet50(), metrics = accuracy)
  summary(learn)
})

test_succeeds('pet load resnet34', {
  learn = cnn_learner(dls, resnet34(), metrics = accuracy)
  summary(learn)
})

test_succeeds('pet load resnet18', {
  learn = cnn_learner(dls, resnet18(), metrics = accuracy)
  summary(learn)
})

test_succeeds('pet load resnet152', {
  learn = cnn_learner(dls, resnet152(), metrics = accuracy)
  summary(learn)
})

test_succeeds('pet load resnet101', {
  learn = cnn_learner(dls, resnet101(), metrics = accuracy)
  summary(learn)
})


test_succeeds('pet load densenet121', {
  learn = cnn_learner(dls, densenet121(), metrics = accuracy)
  summary(learn)
})

test_succeeds('pet load densenet161', {
  learn = cnn_learner(dls, densenet161(), metrics = accuracy)
  summary(learn)
})

test_succeeds('pet load densenet169', {
  learn = cnn_learner(dls, densenet169(), metrics = accuracy)
  summary(learn)
})

test_succeeds('pet load densenet201', {
  learn = cnn_learner(dls, densenet201(), metrics = accuracy)
  summary(learn)
})






