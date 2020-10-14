


#' @title Xresnet50_deep
#'
#' @description Load model architecture
#' @param ... parameters to pass
#' @return model
#' @export
xresnet50_deep <- function(...) {

  args = list(...)
  if(length(args)>0) {
    do.call(vision$all$xresnet50_deep, args)
  } else {
    vision$all$xresnet50_deep
  }
}

#' @title Xresnet50
#'
#' @description Load model architecture
#' @param ... parameters to pass
#' @return model
#' @export
xresnet50 <- function(...) {
  args = list(...)
  if(length(args)>0) {
    do.call(vision$all$xresnet50, args)
  } else {
    vision$all$xresnet50
  }
}

#' @title Xresnet34_deep
#'
#' @description Load model architecture
#' @param ... parameters to pass
#' @return model
#' @export
xresnet34_deep <- function(...) {
  args = list(...)
  if(length(args)>0){
    do.call(vision$all$xresnet34_deep, args)
  } else {
    vision$all$xresnet34_deep
  }
}

#' @title Xresnet34
#'
#' @description Load model architecture
#' @param ... parameters to pass
#' @return model
#' @export
xresnet34 <- function(...) {
  args = list(...)
  if(length(args)>0){
    do.call(vision$all$xresnet34, args)
  } else {
    vision$all$xresnet34
  }
}

#' @title Xresnet18_deep
#'
#' @description Load model architecture
#' @param ... parameters to pass
#' @return model
#' @export
xresnet18_deep <- function(...) {
  args = list(...)
  if(length(args)>0){
    do.call(vision$all$xresnet18_deep, args)
  } else {
    vision$all$xresnet18_deep
  }
}

#' @title Xresnet18
#'
#' @description Load model architecture
#' @param ... parameters to pass
#' @return model
#' @export
xresnet18 <- function(...) {
  args = list(...)
  if(length(args)>0){
    do.call(vision$all$xresnet18, args)
  } else {
    vision$all$xresnet18
  }
}

#' @title Xresnet152
#'
#' @description Load model architecture
#' @param ... parameters to pass
#' @return model
#' @export
xresnet152 <- function(...) {
  args = list(...)
  if(length(args)>0){
    do.call(vision$all$xresnet152, args)
  } else {
    vision$all$xresnet152
  }
}

#' @title Xresnet101
#'
#' @description Load model architecture
#' @param ... parameters to pass
#' @return model
#' @export
xresnet101 <- function(...) {
  args = list(...)
  if(length(args)>0){
    do.call(vision$all$xresnet101, args)
  } else {
    vision$all$xresnet101
  }
}


#' @title Xresnet50_deep
#'
#' @description Load model architecture
#' @param ... parameters to pass
#' @return model
#' @export
xresnet50_deep <- function(...) {
  args = list(...)
  if(length(args)>0) {
    do.call(vision$all$xresnet50_deep, args)
  } else {
    vision$all$xresnet50_deep
  }
}



#' @title Alexnet
#'
#' @description AlexNet model architecture
#'
#' @details "One weird trick..." <https://arxiv.org/abs/1404.5997>
#'
#' @param pretrained pretrained or not
#' @param progress to see progress bar or not
#' @return model
#'
#' @examples
#'
#' \dontrun{
#'
#' alexnet(pretrained = FALSE, progress = TRUE)
#'
#' }
#'
#' @export
alexnet <- function(pretrained = FALSE, progress) {

  if(missing(progress)) {
    vision$all$alexnet
  } else {
    vision$all$alexnet(
      pretrained = pretrained,
      progress = progress
    )
  }

}

#' @title XResNet
#'
#' @description A sequential container.
#'
#' @param block the blocks to pass to XResNet
#' @param expansion argument for inputs and filters
#' @param layers the layers to pass to XResNet
#' @param c_in number of inputs
#' @param c_out number of outputs
#' @param ... additional arguments
#'
#' @export
XResNet <- function(block, expansion, layers, c_in = 3, c_out = 1000,
                    ...) {

  args = list(
    block = block,
    expansion = expansion,
    layers = layers,
    c_in = as.integer(c_in),
    c_out = as.integer(c_out),
    ...
  )

  do.call(vision$all$XResNet, args)

}


#' @title Vgg19_bn
#'
#' @description VGG 19-layer model (configuration 'E') with batch normalization
#'
#' @details "Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>
#'
#' @param pretrained pretrained or not
#' @param progress to see progress bar or not
#' @return model
#' @export
vgg19_bn <- function(pretrained = FALSE, progress) {

  if(missing(progress)) {
    vision$all$vgg19_bn
  } else {
    vision$all$vgg19_bn(
      pretrained = pretrained,
      progress = progress
    )
  }

}

#' @title Vgg16_bn
#'
#' @description VGG 16-layer model (configuration "D") with batch normalization
#'
#' @details "Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>
#'
#' @param pretrained pretrained or not
#' @param progress to see progress bar or not
#' @return model
#'
#' @export
vgg16_bn <- function(pretrained = FALSE, progress) {

  if(missing(progress)) {
    vision$all$vgg16_bn
  } else {
    vision$all$vgg16_bn(
      pretrained = pretrained,
      progress = progress
    )
  }

}

#' @title Vgg13_bn
#'
#' @description VGG 13-layer model (configuration "B") with batch normalization
#'
#' @details "Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>
#'
#' @param pretrained pretrained or not
#' @param progress to see progress bar or not
#' @return model
#'
#' @export
vgg13_bn <- function(pretrained = FALSE, progress) {

  if(missing(progress)) {
    vision$all$vgg13_bn
  } else {
    vision$all$vgg13_bn(
      pretrained = pretrained,
      progress = progress
    )
  }

}

#' @title Vgg11_bn
#'
#' @description VGG 11-layer model (configuration "A") with batch normalization
#'
#' @details "Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>
#'
#' @param pretrained pretrained or not
#' @param progress to see progress bar or not
#' @return model
#'
#' @export
vgg11_bn <- function(pretrained = FALSE, progress) {

  if(missing(progress)) {
    vision$all$vgg11_bn
  } else {
    vision$all$vgg11_bn(
      pretrained = pretrained,
      progress = progress
    )
  }

}

#' @title Squeezenet1_1
#'
#' @description SqueezeNet 1.1 model from the `official SqueezeNet repo
#'
#' @details <https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1>`_.
#' SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters
#' than SqueezeNet 1.0, without sacrificing accuracy.
#'
#' @param pretrained pretrained or not
#' @param progress to see progress bar or not
#' @return model
#'
#' @export
squeezenet1_1 <- function(pretrained = FALSE, progress) {

  if(missing(progress)) {
    vision$all$squeezenet1_1
  } else {
    vision$all$squeezenet1_1(
      pretrained = pretrained,
      progress = progress
    )
  }

}

#' @title Squeezenet1_0
#'
#' @description SqueezeNet model architecture from the `"SqueezeNet: AlexNet-level
#'
#' @details accuracy with 50x fewer parameters and <0.5MB model size"
#' <https://arxiv.org/abs/1602.07360>`_ paper.
#'
#' @param pretrained pretrained or not
#' @param progress to see progress bar or not
#' @return model
#'
#' @export
squeezenet1_0 <- function(pretrained = FALSE, progress) {

  if(missing(progress)) {
    vision$all$squeezenet1_0
  } else {
    vision$all$squeezenet1_0(
      pretrained = pretrained,
      progress = progress
    )
  }

}

#' @title SqueezeNet
#'
#' @description Base class for all neural network modules.
#'
#' @details Your models should also subclass this class. Modules can also contain other Modules, allowing to nest them in
#' a tree structure. You can assign the submodules as regular attributes:: import torch.nn as nn import torch.nn.functional as F class Model(nn.Module): def __init__(self): super(Model, self).__init__() self.conv1 = nn.Conv2d(1, 20, 5) self.conv2 = nn.Conv2d(20, 20, 5) def forward(self, x): x = F.relu(self.conv1(x)) return F.relu(self.conv2(x)) Submodules assigned in this way will be registered, and will have their
#' parameters converted too when you call :meth:`to`, etc.
#'
#' @param version version of SqueezeNet
#' @param num_classes the number of classes
#' @return model
#' @export
SqueezeNet <- function(version = "1_0", num_classes = 1000) {

  vision$all$SqueezeNet(
    version = version,
    num_classes = as.integer(num_classes)
  )

}


#' @title Resnet50
#'
#'
#' @details "Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>
#'
#' @param pretrained pretrained or not
#' @param progress to see progress bar or not
#' @return model
#'
#' @export
resnet50 <- function(pretrained = FALSE, progress) {

  if(missing(progress)) {
    vision$all$resnet50
  } else {
    vision$all$resnet50(
      pretrained = pretrained,
      progress = progress
    )
  }

}

#' @title Resnet34
#'
#' @description ResNet-34 model from
#'
#' @details "Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>
#'
#' @param pretrained pretrained or not
#' @param progress to see progress bar or not
#' @return model
#'
#' @export
resnet34 <- function(pretrained = FALSE, progress) {

  if(missing(progress)) {
    vision$all$resnet34
  } else {
    vision$all$resnet34(
      pretrained = pretrained,
      progress = progress
    )
  }

}

#' @title Resnet18
#'
#'
#' @details "Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>
#'
#' @param pretrained pretrained or not
#' @param progress to see progress bar or not
#' @return model
#'
#' @export
resnet18 <- function(pretrained = FALSE, progress) {

  if(missing(progress)) {
    vision$all$resnet18
  } else {
    vision$all$resnet18(
      pretrained = pretrained,
      progress = progress
    )
  }

}

#' @title Resnet152
#'
#'
#' @details "Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>
#'
#' @param pretrained pretrained or not
#' @param progress to see progress bar or not
#' @return model
#'
#' @export
resnet152 <- function(pretrained = FALSE, progress) {

  if(missing(progress)) {
    vision$all$resnet152
  } else {
    vision$all$resnet152(
      pretrained = pretrained,
      progress = progress
    )
  }

}

#' @title Resnet101
#'
#' @description ResNet-101 model from
#'
#' @details "Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>
#' @param pretrained pretrained or not
#' @param progress to see progress bar or not
#' @return model
#'
#' @export
resnet101 <- function(pretrained = FALSE, progress) {

  if(missing(progress)) {
    vision$all$resnet101
  } else {
    vision$all$resnet101(
      pretrained = pretrained,
      progress = progress
    )
  }

}


#' @title ResNet
#'
#' @description Base class for all neural network modules.
#' @param block the blocks that need to passed to ResNet
#' @param layers the layers to pass to ResNet
#' @param num_classes the number of classes
#' @param zero_init_residual logical, initializer
#' @param groups the groups
#' @param width_per_group the width per group
#' @param replace_stride_with_dilation logical, replace stride with dilation
#' @param norm_layer norm_layer
#'
#' @export
ResNet <- function(block, layers, num_classes = 1000, zero_init_residual = FALSE,
                   groups = 1, width_per_group = 64,
                   replace_stride_with_dilation = NULL, norm_layer = NULL) {

   args = list(
    block = block,
    layers = layers,
    num_classes = as.integer(num_classes),
    zero_init_residual = zero_init_residual,
    groups = as.integer(groups),
    width_per_group = as.integer(width_per_group),
    replace_stride_with_dilation = replace_stride_with_dilation,
    norm_layer = norm_layer
  )

   do.call(vision$all$ResNet, args)

}

#' @title Densenet121
#'
#'
#' @details "Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>
#'
#' @param pretrained pretrained or not
#' @param progress to see progress bar or not
#' @return model
#'
#' @export
densenet121 <- function(pretrained = FALSE, progress) {

  if(missing(progress)) {
    vision$all$densenet121
  } else {
    vision$all$densenet121(
      pretrained = pretrained,
      progress = progress
    )
  }

}

#' @title Densenet161
#'
#'
#' @details "Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>
#'
#' @param pretrained pretrained or not
#' @param progress to see progress bar or not
#' @return model
#'
#' @export
densenet161 <- function(pretrained = FALSE, progress) {

  if(missing(progress)) {
    vision$all$densenet161
  } else {
    vision$all$densenet161(
      pretrained = pretrained,
      progress = progress
    )
  }

}

#' @title Densenet169
#'
#'
#' @details "Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>
#'
#' @param pretrained pretrained or not
#' @param progress to see progress bar or not
#' @return model
#'
#' @export
densenet169 <- function(pretrained = FALSE, progress) {

  if(missing(progress)) {
    vision$all$densenet169
  } else {
    vision$all$densenet169(
      pretrained = pretrained,
      progress = progress
    )
  }

}

#' @title Densenet201
#'
#'
#' @details "Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>
#'
#' @param pretrained pretrained or not
#' @param progress to see progress bar or not
#' @return model
#'
#' @export
densenet201 <- function(pretrained = FALSE, progress) {

  if(missing(progress)) {
    vision$all$densenet201
  } else {
    vision$all$densenet201(
      pretrained = pretrained,
      progress = progress
    )
  }

}






