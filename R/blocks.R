

#' @title SEBlock
#'
#' @param expansion decoder
#' @param ni ni
#' @param nf nf
#' @param groups number of groups
#' @param reduction number of reduction
#' @param stride number of stride
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

#' @title SEResNeXtBlock
#'
#' @param expansion decoder
#' @param ni ni
#' @param nf nf
#' @param groups number of groups
#' @param reduction number of reduction
#' @param stride number of stride
#' @param base_width base width
#' @return Block object
#' @export
SEResNeXtBlock <- function(expansion, ni, nf, groups = 32, reduction = 16, stride = 1, base_width = 4) {

  vision$all$SEResNeXtBlock(
    expansion = expansion,
    ni = ni,
    nf = nf,
    groups = as.integer(),
    reduction = as.integer(reduction),
    stride = as.integer(stride),
    base_width = as.integer(base_width)
  )

}

#' @title SeparableBlock
#'
#'
#' @param expansion decoder
#' @param ni ni
#' @param nf nf
#' @param reduction number of reduction
#' @param stride number of stride
#' @param base_width base width
#' @return Block object
#'
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

#' @title ResBlock
#'
#' @description Resnet block from `ni` to `nh` with `stride`
#'
#'
#' @param expansion decoder
#' @param ni ni
#' @param nf nf
#' @param stride stride number
#' @param groups groups number
#' @param reduction reduction
#' @param nh1 nh1
#' @param nh2 nh2
#' @param dw dw
#' @param g2 g2
#' @param sa sa
#' @param sym sym
#' @param norm_type norm_type
#' @param act_cls act_cls
#' @param ndim dimension number
#' @param ks ks
#' @param pool pooling type, Average, Max
#' @param pool_first pool_first
#' @param padding padding
#' @param bias bias
#' @param bn_1st bn 1st
#' @param transpose transpose
#' @param init initializer
#' @param xtra xtra
#' @param bias_std bias standard deviation
#' @param dilation dilation number
#' @param padding_mode padding mode
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

#' @title MaxPool
#'
#' @description nn.MaxPool layer for `ndim`
#'
#'
#' @param ks ks
#' @param stride stride
#' @param padding padding
#' @param ndim ndim
#' @param ceil_mode ceil_mode
#' @return None
#' @export
MaxPool <- function(ks = 2, stride = NULL, padding = 0, ndim = 2, ceil_mode = FALSE) {

  vision$all$MaxPool(
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
#' @param expansion decoder
#' @param ni ni
#' @param nf nf
#' @param reduction reduction number
#' @param stride stride number
#' @param base_width int, base width
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
#' @param vocab vocabulary
#' @param sort sort or not
#' @param add_na add NA
#' @return Block object
#' @export
CategoryBlock <- function(vocab = NULL, sort = TRUE, add_na = FALSE) {

  if(is.null(vocab)) {
    text$CategoryBlock
  } else {
    text$CategoryBlock(
      vocab = vocab,
      sort = sort,
      add_na = add_na
    )
  }

}



#' @title SEResNeXtBlock
#'
#'
#' @param expansion decoder
#' @param ni ni
#' @param nf nf
#' @param groups groups number
#' @param reduction reduction number
#' @param stride stride number
#' @param base_width int, base width
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
#' @param n_out output shape
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
#' @param encoded encoded or not
#' @param vocab vocabulary
#' @param add_na add NA
#' @return Block object
#' @export
MultiCategoryBlock <- function(encoded = FALSE, vocab = NULL, add_na = FALSE) {

  vision$all$MultiCategoryBlock(
    encoded = encoded,
    vocab = vocab,
    add_na = add_na
  )

}




#' @title DataBlock
#'
#' @description Generic container to quickly build `Datasets` and `DataLoaders`
#'
#'
#' @param blocks blocks
#' @param dl_type DL applications
#' @param getters how to get dataet
#' @param n_inp n_inp is the number of elements in the tuples that should be considered part
#' of the input and will default to 1 if tfms consists of one set of transforms
#' @param item_tfms One or several transforms applied to the items before batching them
#' @param batch_tfms One or several transforms applied to the batches once they are formed
#' @param ... additional parameters to pass
#' @return block
#' @export
DataBlock <- function(blocks = NULL, dl_type = NULL, getters = NULL,
                      n_inp = NULL, item_tfms = NULL, batch_tfms = NULL,
                      ...) {

  args <- list(
    blocks = blocks,
    dl_type = dl_type,
    getters = getters,
    n_inp = n_inp,
    item_tfms = item_tfms,
    batch_tfms = batch_tfms,
    ...
  )

  if(!is.null(args$batch_tfms)) {
    args$batch_tfms <- unlist(args$batch_tfms)
  }

  if(!is.null(args$n_inp)) {
    args$n_inp <- as.integer(args$n_inp)
  }

  do.call(vision$gan$DataBlock, args)

}



#' @title TransformBlock
#'
#' @description A basic wrapper that links defaults transforms for the data block API
#'
#'
#' @param type_tfms transforamtion type
#' @param item_tfms item transofrmation type
#' @param batch_tfms one or several transforms applied to the batches once they are formed
#' @param dl_type DL applications
#' @param dls_kwargs additional argument
#' @return block
#' @export
TransformBlock <- function(type_tfms = NULL, item_tfms = NULL,
                           batch_tfms = NULL, dl_type = NULL,
                           dls_kwargs = NULL) {



  if(missing(type_tfms) & missing(item_tfms) & missing(batch_tfms) & missing(dl_type) & missing(dls_kwargs)) {
    invisible(vision$gan$TransformBlock)
  } else {
    args <- list(
      type_tfms = type_tfms,
      item_tfms = item_tfms,
      batch_tfms = batch_tfms,
      dl_type = dl_type,
      dls_kwargs = dls_kwargs
    )
    do.call(vision$gan$TransformBlock, args)
  }


}


#' @title ImageBlock
#'
#' @description A `TransformBlock` for images of `cls`
#' @param ... parameters to pass
#' @return block
#' @export
ImageBlock <- function(...) {

  invisible(vision$gan$ImageBlock(...))

}





