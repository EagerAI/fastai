#' @title DataBlock
#'
#' @description Generic container to quickly build `Datasets` and `DataLoaders`
#'
#'
#' @param blocks blocks
#' @param dl_type dl_type
#' @param getters getters
#' @param n_inp n_inp is the number of elements in the tuples that should be considered part of the input and will default to 1 if tfms consists of one set of transforms
#' @param item_tfms One or several transforms applied to the items before batching them
#' @param batch_tfms One or several transforms applied to the batches once they are formed
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
#' @param batch_tfms one or several transforms applied to the batches once they are formed
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
#' @param valid_idx The indices to use for the validation set (defaults to a random split otherwise)
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
#' @param n_features The number of features
#' @param n_extra_layers The number of extra layers
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
#' @param n_channels The number of channels
#' @param n_features The number of features
#' @param n_extra_layers The number of extra layers
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
#' @param padding_mode Mode of padding
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
fit.fastai.vision.gan.GANLearner <- function(object, ...) {

  args <- list(
    ...
  )
  if(!is.null(args[[1]]) & is.null(names(args[[1]]))) {
    args[[1]] = as.integer(args[[1]])
  }

  find_epoch = which(names(args)=='n_epoch')

  if(length(find_epoch)>0) {
    args[[find_epoch]] = as.integer(args[[find_epoch]])
  }

  # fit the model
  do.call(object$fit, args)

  if (length(length(object$recorder$values))==1) {
    history = data.frame(values = do.call(rbind,lapply(1:length(object$recorder$values),
                                                       function(x) object$recorder$values[[x]]$items))
    )
  } else {
    history = data.frame(values = t(do.call(rbind,lapply(1:length(object$recorder$values),
                                                         function(x) object$recorder$values[[x]]$items)))
    )
  }

  nm = object$recorder$metric_names$items
  colnames(history) = nm[!nm %in% c('epoch','time')]

  if(nrow(history)==1) {
    history['epoch'] = 0
  } else {
    history['epoch'] = 0:(nrow(history)-1)
  }

  history = history[,c(which(colnames(history)=="epoch"),which(colnames(history)!="epoch"))]
  invisible(history)

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


#' @title AddChannels
#'
#' @description Add `n_dim` channels at the end of the input.
#'
#'
#' @param n_dim n_dim
#'
#' @export
AddChannels <- function(n_dim) {

  vision$gan$AddChannels(
    n_dim = as.integer(n_dim)
  )

}




#' @title DenseResBlock
#'
#' @description Resnet block of `nf` features. `conv_kwargs` are passed to `conv_layer`.
#'
#' @details
#'
#' @param nf nf
#' @param norm_type norm_type
#' @param ks ks
#' @param stride stride
#' @param padding padding
#' @param bias bias
#' @param ndim ndim
#' @param bn_1st bn_1st
#' @param act_cls act_cls
#' @param transpose transpose
#' @param init init
#' @param xtra xtra
#' @param bias_std bias_std
#' @param dilation dilation
#' @param groups groups
#' @param padding_mode padding_mode
#'
#' @export
DenseResBlock <- function(nf, norm_type = 1,
                          ks = 3, stride = 1, padding = NULL,
                          bias = NULL, ndim = 2, bn_1st = TRUE,
                          act_cls = nn$ReLU, transpose = FALSE, init = "auto", xtra = NULL,
                          bias_std = 0.01, dilation = 1, groups = 1,
                          padding_mode = "zeros") {

  args <- list(
    nf = nf,
    norm_type = as.integer(norm_type),
    ks = as.integer(ks),
    stride = as.integer(stride),
    padding = padding,
    bias = bias,
    ndim = as.integer(ndim),
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

  do.call(vision$gan$DenseResBlock, args)

}

#' @title gan_critic
#'
#' @description Critic to train a `GAN`.
#'
#'
#' @param n_channels n_channels
#' @param nf nf
#' @param n_blocks n_blocks
#' @param p p
#'
#' @export
gan_critic <- function(n_channels = 3, nf = 128, n_blocks = 3, p = 0.15) {

  vision$gan$gan_critic(
    n_channels = as.integer(n_channels),
    nf = as.integer(nf),
    n_blocks = as.integer(n_blocks),
    p = p
  )

}


#' @title GANLoss
#'
#' @description Wrapper around `crit_loss_func` and `gen_loss_func`
#'
#'
#' @param gen_loss_func gen_loss_func
#' @param crit_loss_func crit_loss_func
#' @param gan_model gan_model
#'
#' @export
GANLoss <- function(gen_loss_func, crit_loss_func, gan_model) {

  vision$gan$GANLoss(
    gen_loss_func = gen_loss_func,
    crit_loss_func = crit_loss_func,
    gan_model = gan_model
  )

}

#' @title AdaptiveLoss
#'
#' @description Expand the `target` to match the `output` size before applying `crit`.
#'
#'
#' @param crit crit
#'
#' @export
AdaptiveLoss <- function(crit) {

  vision$gan$AdaptiveLoss(
    crit = crit
  )

}


#' @title accuracy_thresh_expand
#'
#' @description Compute accuracy after expanding `y_true` to the size of `y_pred`.
#'
#'
#' @param y_pred y_pred
#' @param y_true y_true
#' @param thresh thresh
#' @param sigmoid sigmoid
#'
#' @export
accuracy_thresh_expand <- function(y_pred, y_true, thresh = 0.5, sigmoid = TRUE) {

  vision$gan$accuracy_thresh_expand(
    y_pred = y_pred,
    y_true = y_true,
    thresh = thresh,
    sigmoid = sigmoid
  )

}

#' @title set_freeze_model
#'
#'
#' @param m m
#' @param rg rg
#'
#' @export
set_freeze_model <- function(m, rg) {

  vision$gan$set_freeze_model(
    m = m,
    rg = rg
  )

}

#' @title GANTrainer
#'
#' @description Handles GAN Training.
#'
#'
#' @param switch_eval switch_eval
#' @param clip clip
#' @param beta beta
#' @param gen_first gen_first
#' @param show_img show_img
#'
#' @export
GANTrainer <- function(switch_eval = FALSE, clip = NULL, beta = 0.98,
                       gen_first = FALSE, show_img = TRUE) {

  vision$gan$GANTrainer(
    switch_eval = switch_eval,
    clip = clip,
    beta = beta,
    gen_first = gen_first,
    show_img = show_img
  )

}


#' @title FixedGANSwitcher
#'
#' @description Switcher to do `n_crit` iterations of the critic then `n_gen` iterations of the generator.
#'
#' @details
#'
#' @param n_crit n_crit
#' @param n_gen n_gen
#'
#' @export
FixedGANSwitcher <- function(n_crit = 1, n_gen = 1) {

  vision$gan$FixedGANSwitcher(
    n_crit = as.integer(n_crit),
    n_gen = as.integer(n_gen)
  )

}

#' @title AdaptiveGANSwitcher
#'
#' @description Switcher that goes back to generator/critic when the loss goes below `gen_thresh`/`crit_thresh`.
#'
#'
#' @param gen_thresh gen_thresh
#' @param critic_thresh critic_thresh
#'
#' @export
AdaptiveGANSwitcher <- function(gen_thresh = NULL, critic_thresh = NULL) {

  vision$gan$AdaptiveGANSwitcher(
    gen_thresh = gen_thresh,
    critic_thresh = critic_thresh
  )

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

#' @title InvisibleTensor
#'
#' @param x x
#'
#' @export
InvisibleTensor <- function(x) {

  vision$gan$InvisibleTensor(
    x = x
  )

}

#' @title gan_loss_from_func
#'
#' @description Define loss functions for a GAN from `loss_gen` and `loss_crit`.
#'
#'
#' @param loss_gen loss_gen
#' @param loss_crit loss_crit
#' @param weights_gen weights_gen
#'
#' @export
gan_loss_from_func <- function(loss_gen, loss_crit, weights_gen = NULL) {

  vision$gan$gan_loss_from_func(
    loss_gen = loss_gen,
    loss_crit = loss_crit,
    weights_gen = weights_gen
  )

}
