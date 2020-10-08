
context("pretrained_model_weights")

source("utils.R")

test_succeeds('can load xresnet50_deep', {
  xresnet50_deep()
})

test_succeeds('can load xresnet50', {
  xresnet50()
})

test_succeeds('can load xresnet34_deep', {
  xresnet34_deep()
})

test_succeeds('can load xresnet34', {
  xresnet34()
})

test_succeeds('can load xresnet18_deep', {
  xresnet34_deep()
})

test_succeeds('can load xresnet18', {
  xresnet18()
})


test_succeeds('can load xresnet152', {
  xresnet152()
})

test_succeeds('can load xresnet101', {
  xresnet101()
})

test_succeeds('can load xresnet50_deep', {
  xresnet50_deep()
})


test_succeeds('can load alexnet', {
  alexnet(progress = TRUE)
})

test_succeeds('can load vgg19_bn', {
  vgg19_bn(progress = TRUE)
})

test_succeeds('can load vgg16_bn', {
  vgg16_bn(progress = TRUE)
})

test_succeeds('can load vgg13_bn', {
  vgg13_bn(progress = TRUE)
})

test_succeeds('can load vgg11_bn', {
  vgg11_bn(progress = TRUE)
})

test_succeeds('can load squeezenet1_1', {
  squeezenet1_1(progress = TRUE)
})

test_succeeds('can load squeezenet1_0', {
  squeezenet1_0(progress = TRUE)
})

test_succeeds('can load SqueezeNet', {
  SqueezeNet()
})

test_succeeds('can load resnet50', {
  resnet50(progress = TRUE)
})

test_succeeds('can load resnet34', {
  resnet34(progress = TRUE)
})

test_succeeds('can load resnet18', {
  resnet18(progress = TRUE)
})

test_succeeds('can load resnet152', {
  resnet152(progress = TRUE)
})

test_succeeds('can load resnet101', {
  resnet101(progress = TRUE)
})

test_succeeds('can load densenet121', {
  densenet121(progress = TRUE)
})

test_succeeds('can load densenet161', {
  densenet161(progress = TRUE)
})

test_succeeds('can load densenet169', {
  densenet169(progress = TRUE)
})

test_succeeds('can load densenet201', {
  densenet201(progress = TRUE)
})









