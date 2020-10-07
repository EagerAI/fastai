

#' @title SEBlock
#'
#' @param expansion expansion
#' @param ni ni
#' @param nf nf
#' @param groups groups
#' @param reduction reduction
#' @param stride stride
#' @return Block object
#' @export
SEBlock <- function(expansion, ni, nf, groups = 1, reduction = 16, stride = 1) {

  vision$all$SEBlock(
    expansion = expansion,
    ni = ni,
    nf = nf,
    groups = as.integer(),
    reduction = as.integer(reduction),
    stride = as.integer(stride)
  )

}

#' @title ResBlock
#'
#' @description Resnet block from `ni` to `nh` with `stride`
#'
#'
#' @param expansion expansion
#' @param ni ni
#' @param nf nf
#' @param stride stride
#' @param groups groups
#' @param reduction reduction
#' @param nh1 nh1
#' @param nh2 nh2
#' @param dw dw
#' @param g2 g2
#' @param sa sa
#' @param sym sym
#' @param norm_type norm_type
#' @param act_cls act_cls
#' @param ndim ndim
#' @param ks ks
#' @param pool pool
#' @param pool_first pool_first
#' @param padding padding
#' @param bias bias
#' @param bn_1st bn_1st
#' @param transpose transpose
#' @param init init
#' @param xtra xtra
#' @param bias_std bias_std
#' @param dilation dilation
#' @param padding_mode padding_mode
#' @return Block object
#' @export
ResBlock <- function(expansion, ni, nf, stride = 1, groups = 1,
                     reduction = NULL, nh1 = NULL, nh2 = NULL, dw = FALSE,
                     g2 = 1, sa = FALSE, sym = FALSE, norm_type = 1,
                     act_cls = nn$ReLU, ndim = 2, ks = 3, pool = AvgPool(), pool_first = TRUE,
                     padding = NULL, bias = NULL, bn_1st = TRUE, transpose = FALSE, init = "auto",
                     xtra = NULL, bias_std = 0.01, dilation = 1, padding_mode = "zeros") {

  vision$all$ResBlock(
    expansion = expansion,
    ni = ni,
    nf = nf,
    stride = as.integer(stride),
    groups = as.integer(groups),
    reduction = reduction,
    nh1 = nh1,
    nh2 = nh2,
    dw = dw,
    g2 = as.integer(g2),
    sa = sa,
    sym = sym,
    norm_type = as.integer(norm_type),
    act_cls = act_cls,
    ndim = as.integer(ndim),
    ks = as.integer(ks),
    pool = pool,
    pool_first = pool_first,
    padding = padding,
    bias = bias,
    bn_1st = bn_1st,
    transpose = transpose,
    init = init,
    xtra = xtra,
    bias_std = bias_std,
    dilation = as.integer(dilation),
    padding_mode = padding_mode
  )

}

#' @title AvgPool
#'
#' @description nn$AvgPool layer for `ndim`
#'
#'
#' @param ks ks
#' @param stride stride
#' @param padding padding
#' @param ndim ndim
#' @param ceil_mode ceil_mode
#' @return None
#' @export
AvgPool <- function(ks = 2, stride = NULL, padding = 0, ndim = 2, ceil_mode = FALSE) {

  vision$all$AvgPool(
    ks = as.integer(ks),
    stride = stride,
    padding = as.integer(padding),
    ndim = as.integer(ndim),
    ceil_mode = ceil_mode
  )

}


#' @title SeparableBlock
#'
#'
#' @param expansion expansion
#' @param ni ni
#' @param nf nf
#' @param reduction reduction
#' @param stride stride
#' @param base_width base_width
#' @return Block object
#' @export
SeparableBlock <- function(expansion, ni, nf, reduction = 16, stride = 1, base_width = 4) {

  vision$all$SeparableBlock(
    expansion = expansion,
    ni = ni,
    nf = nf,
    reduction = as.integer(reduction),
    stride = as.integer(stride),
    base_width = as.integer(base_width)
  )

}


#' @title CategoryBlock
#'
#' @description `TransformBlock` for single-label categorical targets
#'
#'
#' @param vocab vocab
#' @param sort sort
#' @param add_na add_na
#' @return Block object
#' @export
CategoryBlock <- function(vocab = NULL, sort = TRUE, add_na = FALSE) {

  vision$all$CategoryBlock(
    vocab = vocab,
    sort = sort,
    add_na = add_na
  )

}



#' @title SEResNeXtBlock
#'
#'
#' @param expansion expansion
#' @param ni ni
#' @param nf nf
#' @param groups groups
#' @param reduction reduction
#' @param stride stride
#' @param base_width base_width
#' @return Block object
#' @export
SEResNeXtBlock <- function(expansion, ni, nf, groups = 32, reduction = 16, stride = 1, base_width = 4) {

  vision$all$SEResNeXtBlock(
    expansion = expansion,
    ni = ni,
    nf = nf,
    groups = as.integer(groups),
    reduction = as.integer(reduction),
    stride = as.integer(stride),
    base_width = as.integer(base_width)
  )

}



#' @title RegressionBlock
#'
#' @description `TransformBlock` for float targets
#'
#'
#' @param n_out n_out
#' @return Block object
#' @export
RegressionBlock <- function(n_out = NULL) {

  vision$all$RegressionBlock(
    n_out = n_out
  )

}

#' @title MultiCategoryBlock
#'
#' @description `TransformBlock` for multi-label categorical targets
#'
#'
#' @param encoded encoded
#' @param vocab vocab
#' @param add_na add_na
#' @return Block object
#' @export
MultiCategoryBlock <- function(encoded = FALSE, vocab = NULL, add_na = FALSE) {

  vision$all$MultiCategoryBlock(
    encoded = encoded,
    vocab = vocab,
    add_na = add_na
  )

}










