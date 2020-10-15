

#' @title Module
#'
#' @details Decorator to create an nn$Module using f as forward method
#'
#' @param ... parameters to pass
#'
#' @return None
fmodule = function(...) {
  args = list(...)
  do.call(fastai2$vision$all$module, args)
}


#' @title Lambda
#'
#' @description An easy way to create a pytorch layer for a simple `func`
#'
#'
#' @param func function
#' @return None
#' @export
Lambda <- function(func) {

  fastai2$vision$all$Lambda(
    func = func
  )

}


#' @title Partial Lambda
#'
#' @description Layer that applies `partial(func, ...)`
#'
#'
#' @param func function
#' @return None
#' @export
PartialLambda <- function(func) {

  fastai2$vision$all$PartialLambda(
    func = func
  )

}

#' @title Flatten
#'
#' @description Flatten `x` to a single dimension, e.g. at end of a model. `full` for rank-1 tensor
#'
#'
#' @param full bool, full or not
#'
#' @export
Flatten <- function(full = FALSE) {

  fastai2$layers$Flatten(
    full = full
  )

}


#' @title View
#'
#' @description Reshape x to size
#'
#'
#' @param ... parameters to pass
#' @return None
#' @export
View <- function(...) {

  fastai2$layers$View(
    ...
  )

}


#' @title ResizeBatch
#'
#' @description Reshape x to size, keeping batch dim the same size
#'
#'
#' @param ... parameters to pass
#' @return None
#' @export
ResizeBatch <- function(...) {

  fastai2$layers$ResizeBatch(
    ...
  )

}

#' @title Debugger
#'
#' @description A module to debug inside a model
#'
#'
#' @param ... parameters to pass
#' @return None
#' @export
Debugger <- function(...) {

  fastai2$layers$Debugger(
    ...
  )

}



#' @title Sigmoid_range
#'
#' @description Sigmoid function with range `(low, high)`
#'
#'
#' @param x tensor
#' @param low low value
#' @param high high value
#' @return None
#' @export
sigmoid_range <- function(x, low, high) {

  fastai2$layers$sigmoid_range(
    x = x,
    low = low,
    high = high
  )

}



#' @title SigmoidRange
#'
#' @description Sigmoid module with range `(low, high)`
#'
#'
#' @param low low value
#' @param high high value
#' @return None
#' @export
SigmoidRange <- function(low, high) {

  fastai2$layers$SigmoidRange(
    low = low,
    high = high
  )

}

#' @title AdaptiveConcatPool1d
#'
#' @description Layer that concats `AdaptiveAvgPool1d` and `AdaptiveMaxPool1d`
#'
#'
#' @param size output size
#' @return None
#' @export
AdaptiveConcatPool1d <- function(size = NULL) {

  fastai2$layers$AdaptiveConcatPool1d(
    size = as.integer(size)
  )

}

#' @title AdaptiveConcatPool2d
#'
#' @description Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`
#'
#'
#' @param size output size
#' @return None
#' @export
AdaptiveConcatPool2d <- function(size = NULL) {

  fastai2$layers$AdaptiveConcatPool2d(
    size = as.integer(size)
  )

}


#' @title Adaptive_pool
#'
#'
#' @param pool_type pooling type
#' @return Nonee
#' @export
adaptive_pool <- function(pool_type) {

  fastai2$layers$adaptive_pool(
    pool_type = pool_type
  )

}


#' @title PoolFlatten
#'
#' @description Combine `nn.AdaptiveAvgPool2d` and `Flatten`.
#'
#'
#' @param pool_type pooling type
#' @return None
#' @export
PoolFlatten <- function(pool_type = "Avg") {

  fastai2$layers$PoolFlatten(
    pool_type = pool_type
  )

}


#' @title BatchNorm
#'
#' @description BatchNorm layer with `nf` features and `ndim` initialized depending on `norm_type`.
#'
#' @param nf input shape
#' @param ndim dimension number
#' @param norm_type normalization type
#' @param eps epsilon
#' @param momentum momentum
#' @param affine affine
#' @param track_running_stats track running statistics
#' @return None
#' @export
BatchNorm <- function(nf, ndim = 2, norm_type = 1,
                      eps = 1e-05, momentum = 0.1, affine = TRUE,
                      track_running_stats = TRUE) {

  fastai2$layers$BatchNorm(
    nf = as.integer(nf),
    ndim = as.integer(ndim),
    norm_type = as.integer(norm_type),
    eps = eps,
    momentum = momentum,
    affine = affine,
    track_running_stats = track_running_stats
  )

}


#' @title InstanceNorm
#'
#' @description InstanceNorm layer with `nf` features and `ndim` initialized depending on `norm_type`.
#'
#'
#' @param nf input shape
#' @param ndim dimension number
#' @param norm_type normalization type
#' @param eps epsilon
#' @param momentum momentum
#' @param affine affine
#' @param track_running_stats track running statistics
#' @return None
#' @export
InstanceNorm <- function(nf, ndim = 2, norm_type = 5,
                         affine = TRUE, eps = 1e-05, momentum = 0.1,
                         track_running_stats = FALSE) {

  fastai2$layers$InstanceNorm(
    nf = as.integer(nf),
    ndim = as.integer(ndim),
    norm_type = as.integer(norm_type),
    affine = affine,
    eps = eps,
    momentum = momentum,
    track_running_stats = track_running_stats
  )

}


#' @title BatchNorm1dFlat
#'
#' @description `nn.BatchNorm1d`, but first flattens leading dimensions
#'
#'
#' @param num_features number of features
#' @param eps epsilon
#' @param momentum momentum
#' @param affine affine
#' @param track_running_stats track running statistics
#' @return None
#' @export
BatchNorm1dFlat <- function(num_features, eps = 1e-05, momentum = 0.1,
                            affine = TRUE, track_running_stats = TRUE) {

  fastai2$layers$BatchNorm1dFlat(
    num_features = as.integer(num_features),
    eps = eps,
    momentum = momentum,
    affine = affine,
    track_running_stats = track_running_stats
  )

}

#' @title LinBnDrop
#'
#' @description Module grouping `BatchNorm1d`, `Dropout` and `Linear` layers
#
#'
#' @param n_in input shape
#' @param n_out output shape
#' @param bn bn
#' @param p probability
#' @param act activation
#' @param lin_first linear first
#' @return None
#' @export
LinBnDrop <- function(n_in, n_out, bn = TRUE, p = 0.0, act = NULL, lin_first = FALSE) {

  fastai2$layers$LinBnDrop(
    n_in = n_in,
    n_out = n_out,
    bn = bn,
    p = p,
    act = act,
    lin_first = lin_first
  )

}


#' @title Sigmoid
#'
#' @description Same as `torch$sigmoid`, plus clamping to `(eps,1-eps)
#'
#'
#' @param input input
#' @param eps epsilon
#' @return None
#' @export
sigmoid <- function(input, eps = 1e-07) {

  fastai2$layers$sigmoid(
    input = input,
    eps = eps
  )

}

#' @title Sigmoid_
#'
#' @description Same as `torch$sigmoid_`, plus clamping to `(eps,1-eps)
#'
#'
#' @param input input
#' @param eps eps
#' @return None
#' @export
sigmoid_ <- function(input, eps = 1e-07) {

  fastai2$layers$sigmoid_(
    input = input,
    eps = eps
  )

}


#' @title Vleaky_relu
#'
#' @description `F$leaky_relu` with 0.3 slope
#'
#'
#' @param input input
#' @param inplace inplace
#' @return None
#' @export
vleaky_relu <- function(input, inplace = TRUE) {

  fastai2$layers$vleaky_relu(
    input = input,
    inplace = inplace
  )

}



#' @title Init_default
#'
#' @description Initialize `m` weights with `func` and set `bias` to 0.
#'
#'
#' @param m m parameter
#' @param func function
#' @return None
#' @export
init_default <- function(m, func = nn$init$kaiming_normal_) {

  fastai2$layers$init_default(
    m = m,
    func = func
  )

}

#' @title Init_linear
#'
#'
#' @param m m parameter
#' @param act_func activation function
#' @param init initializer
#' @param bias_std bias standard deviation
#' @return None
#' @export
init_linear <- function(m, act_func = NULL, init = "auto", bias_std = 0.01) {

  fastai2$layers$init_linear(
    m = m,
    act_func = act_func,
    init = init,
    bias_std = bias_std
  )

}



####### Layers

#' @title ConvLayer
#'
#' @description Create a sequence of convolutional (`ni` to `nf`), ReLU (if `use_activ`) and `norm_type` layers.
#'
#' @param ni input
#' @param nf output
#' @param ks kernel size
#' @param stride stride
#' @param padding padding
#' @param bias bias
#' @param ndim dimension number
#' @param norm_type normalization type
#' @param bn_1st bn_1st
#' @param act_cls activation
#' @param transpose transpose
#' @param init initializer
#' @param xtra xtra
#' @param bias_std bias standard deviation
#' @param dilation  specify the dilation rate to use for dilated convolution
#' @param groups groups size
#' @param padding_mode padding mode, e.g 'zeros'
#' @return None
#' @export
ConvLayer <- function(ni, nf, ks = 3, stride = 1, padding = NULL, bias = NULL,
                      ndim = 2, norm_type = 1, bn_1st = TRUE,
                      act_cls = nn$ReLU, transpose = FALSE, init = "auto", xtra = NULL,
                      bias_std = 0.01, dilation = 1, groups = 1, padding_mode = "zeros") {

  args = list(
    ni = as.integer(ni),
    nf = as.integer(nf),
    ks = as.integer(ks),
    stride = as.integer(stride),
    padding = padding,
    bias = bias,
    ndim = as.integer(ndim),
    norm_type = as.integer(norm_type),
    bn_1st = bn_1st,
    act_cls = act_cls,
    transpose = transpose,
    init = init,
    xtra = xtra,
    bias_std = bias_std,
    dilation = as.integer(dilation),
    groups = as.integer(groups),
    padding_mode = padding_mode
  )

  do.call(fastai2$layers$ConvLayer, args)

}



#' @title AdaptiveAvgPool
#'
#' @description nn$AdaptiveAvgPool layer for `ndim`
#'
#'
#' @param sz size
#' @param ndim dimension size
#'
#' @export
AdaptiveAvgPool <- function(sz = 1, ndim = 2) {

  fastai2$layers$AdaptiveAvgPool(
    sz = as.integer(sz),
    ndim = as.integer(ndim)
  )

}


#' @title Trunc_normal_
#'
#' @description Truncated normal initialization (approximation)
#'
#'
#' @param x tensor
#' @param mean mean
#' @param std standard deviation
#' @return tensor
#' @export
trunc_normal_ <- function(x, mean = 0.0, std = 1.0) {

  fastai2$layers$trunc_normal_(
    x = x,
    mean = mean,
    std = std
  )

}


#' @title Embedding
#'
#' @description Embedding layer with truncated normal initialization
#'
#'
#' @param ni input
#' @param nf output
#' @return None
#' @export
Embedding <- function(ni, nf) {

  fastai2$layers$Embedding(
    ni = as.integer(ni),
    nf = as.integer(nf)
  )

}


#' @title SelfAttention
#'
#' @description Self attention layer for `n_channels`.
#'
#'
#' @param n_channels number of channels
#' @return None
#' @export
SelfAttention <- function(n_channels) {

  fastai2$layers$SelfAttention(
    n_channels = as.integer(n_channels)
  )

}


#' @title PooledSelfAttention2d
#'
#' @description Pooled self attention layer for 2d.
#'
#'
#' @param n_channels number of channels
#' @return None
#'
#' @export
PooledSelfAttention2d <- function(n_channels) {

  fastai2$layers$PooledSelfAttention2d(
    n_channels = as.integer(n_channels)
  )

}

#' @title SimpleSelfAttention
#'
#' @description Same as `nn$Module`, but no need for subclasses to call `super()$__init__`
#'
#'
#' @param n_in input
#' @param ks kernel size
#' @param sym sym
#' @return None
#' @export
SimpleSelfAttention <- function(n_in, ks = 1, sym = FALSE) {

  fastai2$layers$SimpleSelfAttention(
    n_in = as.integer(n_in),
    ks = as.integer(ks),
    sym = sym
  )

}


#' @title Icnr_init
#'
#' @description ICNR init of `x`, with `scale` and `init` function
#'
#'
#' @param x tensor
#' @param scale int, scale
#' @param init initializer
#' @return None
#' @export
icnr_init <- function(x, scale = 2, init = nn$init$kaiming_normal_) {

  fastai2$layers$icnr_init(
    x = x,
    scale = as.integer(scale),
    init = init
  )

}



#' @title PixelShuffle_ICNR
#'
#' @description Upsample by `scale` from `ni` filters to `nf` (default `ni`), using `nn.PixelShuffle`.
#'
#'
#' @param ni input filters
#' @param nf nf filters
#' @param scale scale
#' @param blur blur
#' @param norm_type normalziation type
#' @param act_cls activation
#' @return None
#' @export
PixelShuffle_ICNR <- function(ni, nf = NULL, scale = 2, blur = FALSE,
                              norm_type = 3, act_cls = nn$ReLU) {

   fastai2$layers$PixelShuffle_ICNR(
    ni = as.integer(ni),
    nf = nf,
    scale = as.integer(scale),
    blur = blur,
    norm_type = as.integer(norm_type),
    act_cls = act_cls
  )

}




#' @title Sequential
#'
#'
#'
#' @param ... parameters to pass
#'
#' @return None
sequential = function(...) {
  args = list(...)

  do.call(fastai2$layers$sequential, args)
}

#' @title SequentialEx
#'
#'
#'
#' @param ... parameters to pass
#'
#' @return None
SequentialEx = function(...) {
  args = list(...)

  do.call(fastai2$layers$SequentialEx, args)
}



#' @title MergeLayer
#'
#' @description Merge a shortcut with the result of the module by adding them or concatenating them if `dense=TRUE`.
#'
#'
#' @param dense dense
#' @return None
#' @export
MergeLayer <- function(dense = FALSE) {

  fastai2$layers$MergeLayer(
    dense = dense
  )

}


#' @title Cat
#'
#' @description Concatenate layers outputs over a given dim
#'
#'
#' @param layers layers
#' @param dim dimension size
#' @return None
#' @export
Cat <- function(layers, dim = 1) {

  fastai2$layers$Cat(
    layers = layers,
    dim = as.integer(dim)
  )

}


#' @title SimpleCNN
#'
#' @description Create a simple CNN with `filters`.
#'
#'
#' @param filters filters number
#' @param kernel_szs kernel size
#' @param strides strides
#' @param bn bn
#' @return None
#' @export
SimpleCNN <- function(filters, kernel_szs = NULL, strides = NULL, bn = TRUE) {

  args = list(
    filters = filters,
    kernel_szs = kernel_szs,
    strides = strides,
    bn = bn
  )

  do.call(fastai2$layers$SimpleCNN, args)

}



#' @title SEModule
#'
#' @param ch ch
#' @param reduction reduction
#' @param act_cls activation
#' @return None
#' @export
SEModule <- function(ch, reduction, act_cls = nn$ReLU) {

  fastai2$layers$SEModule(
    ch = ch,
    reduction = reduction,
    act_cls = act_cls
  )

}


#' @title Swish
#'
#'
#' @param x tensor
#' @param inplace inplace or not
#' @return None
#' @export
swish <- function(x, inplace = FALSE) {

  fastai2$layers$swish(
    x = x,
    inplace = inplace
  )

}

#' @title Swish
#'
#' @description Same as nn$Module, but no need for subclasses to call super()$__init__
#' @param ... parameters to pass
#' @return None
#' @export
Swish_ <- function(...) {

  fastai2$layers$Swish(
    ...
  )

}




#' @title MishJitAutoFn
#'
#' @description Records operation history and defines formulas for differentiating ops.
#' @param ... parameters to pass
#' @return None
#' @export
MishJitAutoFn <- function(...) {

  fastai2$layers$MishJitAutoFn(
    ...
  )

}












