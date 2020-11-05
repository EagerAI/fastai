


#' @title Create_mlp
#'
#' @description A simple model builder to create a bunch of BatchNorm1d, Dropout and
#' Linear layers, with ```act_fn``` activations.
#' @param ni number of input channels
#' @param nout output shape
#' @param linear_sizes linear output sizes
#' @return model
#' @export
create_mlp <- function(ni, nout, linear_sizes = c(500, 500, 500)) {

  tms()$models$create_mlp(
    ni = as.integer(ni),
    nout = as.integer(nout),
    linear_sizes = as.integer(linear_sizes)
  )

}


#' @title Create_fcn
#'
#' @description A bunch of convolutions stacked together.
#'
#'
#' @param ni number of input channels
#' @param nout output shape
#' @param ks kernel size
#' @param conv_sizes convolution sizes
#' @param stride stride
#' @return model
#' @export
create_fcn <- function(ni, nout, ks = 9, conv_sizes = c(128, 256, 128), stride = 1) {

  tms()$models$create_fcn(
    ni = ni,
    nout = nout,
    ks = as.integer(ks),
    conv_sizes = as.integer(conv_sizes),
    stride = as.integer(stride)
  )

}


#' @title Res_block_1d
#'
#' @description Resnet block as described in the paper.
#'
#'
#' @param nf number of features
#' @param ks kernel size
#' @return block
#' @export
res_block_1d <- function(nf, ks = c(5, 3)) {

  tms()$models$res_block_1d(
    nf = as.integer(nf),
    ks = as.integer(ks)
  )

}


#' @title Create_resnet
#'
#' @description Basic 11 Layer - 1D resnet builder
#'
#'
#' @param ni number of input channels
#' @param nout output shape
#' @param kss kernel size
#' @param conv_sizes convolution sizes
#' @param stride stride
#' @return model
#' @export
create_resnet <- function(ni, nout, kss = c(9, 5, 3), conv_sizes = c(64, 128, 128), stride = 1) {

  tms()$models$create_resnet(
    ni = as.integer(ni),
    nout = as.integer(nout),
    kss = as.integer(kss),
    conv_sizes = as.integer(conv_sizes),
    stride = as.integer(stride)
  )

}


#' @title InceptionModule
#'
#' @description The inception Module from `ni` inputs to len('kss')*`nb_filters`+`bottleneck_size`
#'
#'
#' @param ni number of input channels
#' @param nb_filters the number of filters
#' @param kss kernel size
#' @param bottleneck_size bottleneck size
#' @param stride stride
#' @return module
#' @export
InceptionModule <- function(ni, nb_filters = 32, kss = c(39, 19, 9), bottleneck_size = 32, stride = 1) {

  tms()$models$InceptionModule(
    ni = as.integer(ni),
    nb_filters = as.integer(nb_filters),
    kss = as.integer(kss),
    bottleneck_size = as.integer(bottleneck_size),
    stride = as.integer(stride)
  )

}


#' @title Shortcut
#'
#' @description Merge a shortcut with the result of the module by adding them. Adds Conv, BN and ReLU
#'
#'
#' @param ni number of input channels
#' @param nf number of features
#' @param act_fn activation
#' @return None
#' @export
Shortcut <- function(ni, nf, act_fn = nn$ReLU(inplace = TRUE)) {

  tms()$models$Shortcut(
    ni = as.integer(ni),
    nf = as.integer(nf),
    act_fn = act_fn
  )

}


#' @title Create_inception
#'
#' @description Creates an InceptionTime arch from `ni` channels to `nout` outputs.
#'
#'
#' @param ni number of input channels
#' @param nout number of outputs, should be equal to the number of classes for classification tasks.
#' @param kss kernel sizes for the inception Block.
#' @param depth depth
#' @param bottleneck_size The number of channels on the convolution bottleneck.
#' @param nb_filters Channels on the convolution of each kernel.
#' @param head TRUE if we want a head attached.
#' @return model
#' @export
create_inception <- function(ni, nout, kss = c(39, 19, 9), depth = 6,
                             bottleneck_size = 32, nb_filters = 32, head = TRUE) {

  tms()$models$create_inception(
    ni = as.integer(ni),
    nout = as.integer(nout),
    kss = as.integer(kss),
    depth = as.integer(depth),
    bottleneck_size = as.integer(bottleneck_size),
    nb_filters = as.integer(nb_filters),
    head = head
  )

}


