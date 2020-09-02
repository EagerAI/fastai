#' @title DataBlock
#'
#' @description Generic container to quickly build `Datasets` and `DataLoaders`
#'
#'
#' @param blocks blocks
#' @param dl_type dl_type
#' @param getters getters
#' @param n_inp n_inp
#' @param item_tfms item_tfms
#' @param batch_tfms batch_tfms
#'
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

  do.call(vision$gan$DataBlock, args)

}



#' @title TransformBlock
#'
#' @description A basic wrapper that links defaults transforms for the data block API
#'
#'
#' @param type_tfms type_tfms
#' @param item_tfms item_tfms
#' @param batch_tfms batch_tfms
#' @param dl_type dl_type
#' @param dls_kwargs dls_kwargs
#'
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
#'
#'
#' @export
ImageBlock <- function() {

  invisible(vision$gan$ImageBlock)

}

#' @title Generate_noise
#'
#'
#' @param fn fn
#' @param size size
#'
#' @export
generate_noise <- function(fn, size = 100) {



  if(missing(fn)) {
    invisible(vision$gan$generate_noise)
  } else {
    args <- list(
      fn = fn,
      size = as.integer(size)
    )
    do.call(vision$gan$generate_noise,args)
  }


}

#' @title IndexSplitter
#'
#' @description Split `items` so that `val_idx` are in the validation set and the others in the training set
#'
#'
#' @param valid_idx valid_idx
#'
#' @export
IndexSplitter <- function(valid_idx) {

  if(missing(valid_idx)) {
    invisible(vision$gan$IndexSplitter)
  } else {
    args <- list(
      valid_idx = valid_idx
    )
    do.call(vision$gan$IndexSplitter,args)
  }


}


#' @title FileSplitter
#'
#' @description Split `items` by providing file `fname` (contains names of valid items separated by newline).
#'
#'
#' @param fname fname
#'
#' @export
FileSplitter <- function(fname) {

  if(missing(fname)) {
    vision$gan$FileSplitter
  } else {
    vision$gan$FileSplitter(
      fname = fname
    )
  }

}


#' @title dataloaders
#'
#' @description Create a `DataLoaders` object from `source`
#'
#'
#' @param source source
#' @param ... additional parameters to pass
#'
#' @export
dataloaders <- function(object, ...) {

  my_list <- list(
    source = source,
    ...
  )
  for (i in 1:length(my_list)) {
    if(names(my_list)[[i]]=='bs') {
      my_list[['bs']] = as.integer(my_list[['bs']])
    } else if (names(my_list)[[i]]=='batch_size') {
      my_list[['batch_size']] = as.integer(my_list[['batch_size']])
    } else if (names(my_list)[[i]]=='seq_len') {
      my_list[['seq_len']] = as.integer(my_list[['seq_len']])
    }
  }

  do.call(object$dataloaders,my_list)
}


#' @title Basic_generator
#'
#' @description A basic generator from `in_sz` to images `n_channels` x `out_size` x `out_size`.
#'
#'
#' @param out_size out_size
#' @param n_channels n_channels
#' @param in_sz in_sz
#' @param n_features n_features
#' @param n_extra_layers n_extra_layers
#' @param ... additional params to pass
#' @param bias bias
#' @param ndim ndim
#' @param norm_type norm_type
#' @param bn_1st bn_1st
#' @param act_cls act_cls
#' @param transpose transpose
#' @param init init
#' @param xtra xtra
#' @param bias_std bias_std
#' @param dilation dilation
#' @param groups groups
#'
#' @export
basic_generator <- function(out_size, n_channels, in_sz = 100,
                            n_features = 64, n_extra_layers = 0,
                            bias = NULL, ndim = 2,
                            norm_type = 1, bn_1st = TRUE,
                            act_cls = nn$ReLU, init = "auto",
                            xtra = NULL, bias_std = 0.01, dilation = 1,
                            groups = 1,
                            ...) {

  args <- list(
    out_size = out_size,
    n_channels = as.integer(n_channels),
    in_sz = as.integer(in_sz),
    n_features = as.integer(n_features),
    n_extra_layers = as.integer(n_extra_layers),
    bias = bias,
    ndim = as.integer(ndim),
    norm_type = as.integer(norm_type),
    bn_1st = bn_1st,
    act_cls = act_cls,
    init = init,
    xtra = xtra,
    bias_std = bias_std,
    dilation = as.integer(dilation),
    groups = as.integer(groups),
    ...
  )

  do.call(vision$gan$basic_generator, args)
}


#' @title Basic_critic
#'
#' @description A basic critic for images `n_channels` x `in_size` x `in_size`.
#'
#'
#' @param in_size in_size
#' @param n_channels n_channels
#' @param n_features n_features
#' @param n_extra_layers n_extra_layers
#' @param norm_type norm_type
#' @param bias bias
#' @param ndim ndim
#' @param bn_1st bn_1st
#' @param act_cls act_cls
#' @param transpose transpose
#' @param xtra xtra
#' @param bias_std bias_std
#' @param dilation dilation
#' @param groups groups
#' @param padding_mode padding_mode
#' @param ... additional parameters to pass
#'
#' @export
basic_critic <- function(in_size, n_channels, n_features = 64,
                         n_extra_layers = 0, norm_type = 1,
                         bias = NULL,
                         ndim = 2, bn_1st = TRUE, act_cls = nn$ReLU,
                         transpose = FALSE,
                         xtra = NULL, bias_std = 0.01, dilation = 1,
                         groups = 1, padding_mode = "zeros",
                         ...) {

  args <- list(
    in_size = in_size,
    n_channels = as.integer(n_channels),
    n_features = as.integer(n_features),
    n_extra_layers = as.integer(n_extra_layers),
    norm_type = as.integer(norm_type),
    bias = bias,
    ndim = as.integer(ndim),
    bn_1st = bn_1st,
    act_cls = act_cls,
    transpose = transpose,
    xtra = xtra,
    bias_std = bias_std,
    dilation = as.integer(dilation),
    groups = as.integer(groups),
    padding_mode = padding_mode,
    ...
  )

  do.call(vision$gan$basic_critic, args)

}



#' @title Wgan
#'
#' @description Create a WGAN from `data`, `generator` and `critic`.
#'
#' @param dls dls
#' @param generator generator
#' @param critic critic
#' @param switcher switcher
#' @param clip clip
#' @param switch_eval switch_eval
#' @param gen_first gen_first
#' @param show_img show_img
#' @param cbs cbs
#' @param metrics metrics
#' @param opt_func opt_func
#' @param lr lr
#' @param splitter splitter
#' @param path path
#' @param model_dir model_dir
#' @param wd wd
#' @param wd_bn_bias wd_bn_bias
#' @param train_bn train_bn
#' @param moms moms
#'
#' @export
GANLearner_wgan <- function(dls, generator, critic, switcher = NULL, clip = 0.01,
                 switch_eval = FALSE, gen_first = FALSE, show_img = TRUE,
                 cbs = NULL, metrics = NULL,  opt_func = Adam(),
                 lr = 0.001, splitter = trainable_params, path = NULL,
                 model_dir = "models", wd = NULL, wd_bn_bias = FALSE,
                 train_bn = TRUE, moms = list(0.95, 0.85, 0.95)) {

  args <- list(
    dls = dls,
    generator = generator,
    critic = critic,
    switcher = switcher,
    clip = clip,
    switch_eval = switch_eval,
    gen_first = gen_first,
    show_img = show_img,
    cbs = cbs,
    metrics = metrics,
    opt_func = opt_func,
    lr = lr,
    splitter = splitter,
    path = path,
    model_dir = model_dir,
    wd = wd,
    wd_bn_bias = wd_bn_bias,
    train_bn = train_bn,
    moms = moms
  )

  do.call(vision$gan$GANLearner$wgan, args)

}




#' @title Fit
#' @description Fit the model on this learner with `lr` learning rate, `wd` weight decay for `epochs` with `callbacks`.
#'
#' @param epochs epochs
#' @param lr lr
#' @param wd wd
#' @param callbacks callbacks
#'
#' @export
fit.fastai.vision.gan.GANLearner <- function(object, n_epoch, lr = 1e-2, wd = NULL, callbacks = NULL) {

  args <- list(
    n_epoch = as.integer(n_epoch),
    lr = lr,
    wd = wd,
    callbacks = callbacks
  )

  # fit the model
  do.call(object$fit, args)

}



#' @title GANModule
#'
#' @description Wrapper around a `generator` and a `critic` to create a GAN.
#'
#'
#' @param generator generator
#' @param critic critic
#' @param gen_mode gen_mode
#'
#' @export
GANModule <- function(generator = NULL, critic = NULL, gen_mode = FALSE) {

  args <- list(
    generator = generator,
    critic = critic,
    gen_mode = gen_mode
  )

  do.call(vision$gan$GANModule, args)

}


#' @title GANDiscriminativeLR
#'
#' @description `Callback` that handles multiplying the learning rate by `mult_lr` for the critic.
#'
#'
#' @param mult_lr mult_lr
#'
#' @export
GANDiscriminativeLR <- function(mult_lr = 5.0) {

  vision$gan$GANDiscriminativeLR(
    mult_lr = mult_lr
  )

}

#' @title MaskBlock
#'
#' @description A `TransformBlock` for segmentation masks, potentially with `codes`
#'
#'
#' @param codes codes
#'
#' @export
MaskBlock <- function(codes = NULL) {

  if(is.null(codes)) {
    vision$all$MaskBlock
  } else {
    vision$all$MaskBlock(
      codes = codes
    )
  }

}





