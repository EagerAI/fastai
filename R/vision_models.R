


#' @title Model
#'
#' @description Load model architecture
#' @export
xresnet50_deep <- function(...) {
  vision$models$xresnet50_deep(...)
}

#' @title Model
#'
#' @description Load model architecture
#' @export
xresnet50 <- function(...) {
  vision$models$xresnet50(...)
}

#' @title Model
#'
#' @description Load model architecture
#' @export
xresnet34_deep <- function(...) {
  vision$models$xresnet34_deep(...)
}

#' @title Model
#'
#' @description Load model architecture
#' @export
xresnet34<- function(...) {
  vision$models$xresnet34(...)
}

#' @title Model
#'
#' @description Load model architecture
#' @export
xresnet18_deep <- function(...) {
  vision$models$xresnet18_deep(...)
}

#' @title Model
#'
#' @description Load model architecture
#' @export
xresnet18 <- function(...) {
  vision$models$xresnet18(...)
}

#' @title Model
#'
#' @description Load model architecture
#' @export
xresnet152 <- function(...) {
  vision$models$xresnet152(...)
}

#' @title Model
#'
#' @description Load model architecture
#' @export
xresnet101 <- function(...) {
  vision$models$xresnet101(...)
}

#' @title Model
#'
#' @description Load model architecture
#' @export
wrn_22 <- function(...) {
  vision$models$wrn_22(...)
}

#' @title Model
#'
#' @description Load model architecture
#' @export
WideResNet <- function(...) {
  vision$models$WideResNet(...)
}

#' @title Model
#'
#' @description Load model architecture
#' @export
UnetBlock <- function(...) {
  vision$models$UnetBlock(...)
}

#' @title Model
#'
#' @description Load model architecture
#' @export
ResLayer <- function(...) {
  vision$models$ResLayer(...)
}

#' @title Model
#'
#' @description Load model architecture
#' @export
xresnet50_deep <- function(...) {
  vision$models$xresnet50_deep(...)
}

#' @title Model
#'
#' @description Load model architecture
#' @export
DynamicUnet <- function(...) {
  vision$models$DynamicUnet(...)
}


#' @title alexnet
#'
#' @description AlexNet model architecture from the
#'
#' @details `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
#'
#' @param pretrained pretrained
#' @param progress progress
#'
#' @export
alexnet <- function(pretrained = FALSE, progress = TRUE) {

  vision$models$alexnet(
    pretrained = pretrained,
    progress = progress
  )

}

#' @title XResNet
#'
#' @description A sequential container.
#'
#' @details Modules will be added to it in the order they are passed in the constructor.
#' Alternatively, an ordered dict of modules can also be passed in. To make it easier to understand, here is a small example:: # Example of using Sequential model = nn.Sequential( nn.Conv2d(1,20,5), nn.ReLU(), nn.Conv2d(20,64,5), nn.ReLU() ) # Example of using Sequential with OrderedDict model = nn.Sequential(OrderedDict([ ('conv1', nn.Conv2d(1,20,5)), ('relu1', nn.ReLU()), ('conv2', nn.Conv2d(20,64,5)), ('relu2', nn.ReLU()) ]))
#'
#' @param expansion expansion
#' @param layers layers
#' @param c_in c_in
#' @param c_out c_out
#'
#' @export
XResNet <- function(expansion, layers, c_in = 3, c_out = 1000) {

  vision$models$XResNet(
    expansion = expansion,
    layers = layers,
    c_in = as.integer(c_in),
    c_out = asintec_out
  )

}

#' @title xception
#'
#' @description Preview version of Xception network. Not tested yet - use at own risk. No pretrained model yet.
#'
#' @details
#'
#' @param c c
#' @param k k
#' @param n_middle n_middle
#'
#' @export
xception <- function(c, k = 8, n_middle = 8) {

  vision$models$xception(
    c = c,
    k = as.integer(k),
    n_middle = as.integer(n_middle)
  )

}

#' @title vgg19_bn
#'
#' @description VGG 19-layer model (configuration 'E') with batch normalization
#'
#' @details `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
#'
#' @param pretrained pretrained
#' @param progress progress
#'
#' @export
vgg19_bn <- function(pretrained = FALSE, progress = TRUE) {

  vision$models$vgg19_bn(
    pretrained = pretrained,
    progress = progress
  )

}

#' @title vgg16_bn
#'
#' @description VGG 16-layer model (configuration "D") with batch normalization
#'
#' @details `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
#'
#' @param pretrained pretrained
#' @param progress progress
#'
#' @export
vgg16_bn <- function(pretrained = FALSE, progress = TRUE) {

  vision$models$vgg16_bn(
    pretrained = pretrained,
    progress = progress
  )

}

#' @title vgg13_bn
#'
#' @description VGG 13-layer model (configuration "B") with batch normalization
#'
#' @details `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
#'
#' @param pretrained pretrained
#' @param progress progress
#'
#' @export
vgg13_bn <- function(pretrained = FALSE, progress = TRUE) {

  vision$models$vgg13_bn(
    pretrained = pretrained,
    progress = progress
  )

}

#' @title vgg11_bn
#'
#' @description VGG 11-layer model (configuration "A") with batch normalization
#'
#' @details `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
#'
#' @param pretrained pretrained
#' @param progress progress
#'
#' @export
vgg11_bn <- function(pretrained = FALSE, progress = TRUE) {

  vision$models$vgg11_bn(
    pretrained = pretrained,
    progress = progress
  )

}

#' @title squeezenet1_1
#'
#' @description SqueezeNet 1.1 model from the `official SqueezeNet repo
#'
#' @details <https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1>`_.
#' SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters
#' than SqueezeNet 1.0, without sacrificing accuracy.
#'
#' @param pretrained pretrained
#' @param progress progress
#'
#' @export
squeezenet1_1 <- function(pretrained = FALSE, progress = TRUE) {

  vision$models$squeezenet1_1(
    pretrained = pretrained,
    progress = progress
  )

}

#' @title squeezenet1_0
#'
#' @description SqueezeNet model architecture from the `"SqueezeNet: AlexNet-level
#'
#' @details accuracy with 50x fewer parameters and <0.5MB model size"
#' <https://arxiv.org/abs/1602.07360>`_ paper.
#'
#' @param pretrained pretrained
#' @param progress progress
#'
#' @export
squeezenet1_0 <- function(pretrained = FALSE, progress = TRUE) {

  vision$models$squeezenet1_0(
    pretrained = pretrained,
    progress = progress
  )

}

#' @title SqueezeNet
#'
#' @description Base class for all neural network modules.
#'
#' @details Your models should also subclass this class. Modules can also contain other Modules, allowing to nest them in
#' a tree structure. You can assign the submodules as regular attributes:: import torch.nn as nn import torch.nn.functional as F class Model(nn.Module): def __init__(self): super(Model, self).__init__() self.conv1 = nn.Conv2d(1, 20, 5) self.conv2 = nn.Conv2d(20, 20, 5) def forward(self, x): x = F.relu(self.conv1(x)) return F.relu(self.conv2(x)) Submodules assigned in this way will be registered, and will have their
#' parameters converted too when you call :meth:`to`, etc.
#'
#' @param version version
#' @param num_classes num_classes
#'
#' @export
SqueezeNet <- function(version = "1_0", num_classes = 1000) {

  python_function_result <- vision$models$SqueezeNet(
    version = version,
    num_classes = as.integer(num_classes)
  )

}


#' @title resnet50
#'
#' @description ResNet-50 model from
#'
#' @details `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
#'
#' @param pretrained pretrained
#' @param progress progress
#'
#' @export
resnet50 <- function(pretrained = FALSE, progress = TRUE) {

  vision$models$resnet50(
    pretrained = pretrained,
    progress = progress
  )

}

#' @title resnet34
#'
#' @description ResNet-34 model from
#'
#' @details `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
#'
#' @param pretrained pretrained
#' @param progress progress
#'
#' @export
resnet34 <- function(pretrained = FALSE, progress = TRUE) {

  vision$models$resnet34(
    pretrained = pretrained,
    progress = progress
  )

}

#' @title resnet18
#'
#' @description ResNet-18 model from
#'
#' @details `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
#'
#' @param pretrained pretrained
#' @param progress progress
#'
#' @export
resnet18 <- function(pretrained = FALSE, progress = TRUE) {

  vision$models$resnet18(
    pretrained = pretrained,
    progress = progress
  )

}

#' @title resnet152
#'
#' @description ResNet-152 model from
#'
#' @details `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
#'
#' @param pretrained pretrained
#' @param progress progress
#'
#' @export
resnet152 <- function(pretrained = FALSE, progress = TRUE) {

  vision$models$resnet152(
    pretrained = pretrained,
    progress = progress
  )

}

#' @title resnet101
#'
#' @description ResNet-101 model from
#'
#' @details `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
#'
#' @param pretrained pretrained
#' @param progress progress
#'
#' @export
resnet101 <- function(pretrained = FALSE, progress = TRUE) {

  vision$models$resnet101(
    pretrained = pretrained,
    progress = progress
  )

}


#' @title ResNet
#'
#' @description Base class for all neural network modules.
#'
#' @details Your models should also subclass this class. Modules can also contain other Modules, allowing to nest them in
#' a tree structure. You can assign the submodules as regular attributes:: import torch.nn as nn import torch.nn.functional as F class Model(nn.Module): def __init__(self): super(Model, self).__init__() self.conv1 = nn.Conv2d(1, 20, 5) self.conv2 = nn.Conv2d(20, 20, 5) def forward(self, x): x = F.relu(self.conv1(x)) return F.relu(self.conv2(x)) Submodules assigned in this way will be registered, and will have their
#' parameters converted too when you call :meth:`to`, etc.
#'
#' @param block block
#' @param layers layers
#' @param num_classes num_classes
#' @param zero_init_residual zero_init_residual
#' @param groups groups
#' @param width_per_group width_per_group
#' @param replace_stride_with_dilation replace_stride_with_dilation
#' @param norm_layer norm_layer
#'
#' @export
ResNet <- function(block, layers, num_classes = 1000, zero_init_residual = FALSE,
                   groups = 1, width_per_group = 64,
                   replace_stride_with_dilation = NULL, norm_layer = NULL) {

   vision$models$ResNet(
    block = block,
    layers = layers,
    num_classes = as.integer(num_classes),
    zero_init_residual = zero_init_residual,
    groups = as.integer(groups),
    width_per_group = as.integer(width_per_group),
    replace_stride_with_dilation = replace_stride_with_dilation,
    norm_layer = norm_layer
  )

}

#' @title densenet121
#'
#' @description Densenet-121 model from
#'
#' @details `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
#'
#' @param pretrained pretrained
#' @param progress progress
#'
#' @export
densenet121 <- function(pretrained = FALSE, progress = TRUE) {

  vision$models$densenet121(
    pretrained = pretrained,
    progress = progress
  )

}

#' @title densenet161
#'
#' @description Densenet-161 model from
#'
#' @details `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
#'
#' @param pretrained pretrained
#' @param progress progress
#'
#' @export
densenet161 <- function(pretrained = FALSE, progress = TRUE) {

  vision$models$densenet161(
    pretrained = pretrained,
    progress = progress
  )

}

#' @title densenet169
#'
#' @description Densenet-169 model from
#'
#' @details `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
#'
#' @param pretrained pretrained
#' @param progress progress
#'
#' @export
densenet169 <- function(pretrained = FALSE, progress = TRUE) {

  vision$models$densenet169(
    pretrained = pretrained,
    progress = progress
  )

}

#' @title densenet201
#'
#' @description Densenet-201 model from
#'
#' @details `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
#'
#' @param pretrained pretrained
#' @param progress progress
#'
#' @export
densenet201 <- function(pretrained = FALSE, progress = TRUE) {

  vision$models$densenet201(
    pretrained = pretrained,
    progress = progress
  )

}

#' @title mobilenet_v2
#'
#' @description Constructs a MobileNetV2 architecture from
#'
#' @details `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.
#'
#' @param pretrained pretrained
#' @param progress progress
#'
#' @export
mobilenet_v2 <- function(pretrained = FALSE, progress = TRUE) {

 vision$models$mobilenet_v2(
    pretrained = pretrained,
    progress = progress
  )

}









