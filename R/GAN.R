
#' @title Generate noise
#'
#'
#' @param fn path
#' @param size the size
#' @return None
#'
#'
#' @examples
#' \dontrun{
#'
#' generate_noise()
#'
#' }
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

#' @title Index Splitter
#'
#' @description Split `items` so that `val_idx` are in the validation set and the others in the training set
#'
#'
#' @param valid_idx The indices to use for the validation set (defaults to a random split otherwise)
#' @return None
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


#' @title File Splitter
#'
#' @description Split `items` by providing file `fname` (contains names of valid items separated by newline).
#'
#'
#' @param fname file name
#' @return None
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


#' @title Dataloaders from dls object
#'
#' @description Create a `DataLoaders` object from `source`
#'
#' @param object model
#' @param ... additional parameters to pass
#'
#'
#' @examples
#' \dontrun{
#'
#' dls = TabularDataTable(df, procs, cat_names, cont_names,
#' y_names = dep_var, splits = list(tr_idx, ts_idx) ) %>%
#'   dataloaders(bs = 50)
#'
#' }
#'
#' @export
dataloaders <- function(object, ...) {

  my_list <- list(
    ...
  )

  if(!is.null(my_list[['bs']])) {
    my_list[['bs']] = as.integer(my_list[['bs']])
  }

  if(!is.null(my_list[['batch_size']])) {
    my_list[['batch_size']] = as.integer(my_list[['batch_size']])
  }

  if(!is.null(my_list[['seq_len']])) {
    my_list[['seq_len']] = as.integer(my_list[['seq_len']])
  }

  do.call(object$dataloaders,my_list)
}


#' @title Basic generator
#'
#' @description A basic generator from `in_sz` to images `n_channels` x `out_size` x `out_size`.
#'
#'
#' @param out_size out_size
#' @param n_channels n_channels
#' @param ... additional params to pass
#' @return generator object
#'
#' @examples
#' \dontrun{
#'
#' generator = basic_generator(out_size = 64, n_channels = 3, n_extra_layers = 1)
#'
#' }
#'
#' @export
basic_generator <- function(out_size, n_channels,
                            ...) {

  args <- list(
    out_size = out_size,
    n_channels = as.integer(n_channels),
    ...
  )

  strings = c('in_size',
              'in_sz',
              'out_size',
              'n_channels',
              'n_features',
              'n_extra_layers',
              'norm_type',
              'ks',
              'stride',
              'ndim',
              'dilation',
              'groups')

  for (i in 1:length(strings)) {
    if(!is.null(args[[strings[i]]])) {
      args[[strings[i]]] = as.integer(args[[strings[i]]])
    }
  }


  do.call(vision$gan$basic_generator, args)
}


#' @title Basic critic
#'
#' @description A basic critic for images `n_channels` x `in_size` x `in_size`.
#'
#'
#' @param in_size input size
#' @param n_channels The number of channels
#' @param ... additional parameters to pass
#' @return None
#'
#' @examples
#' \dontrun{
#'
#' critic    = basic_critic(in_size = 64, n_channels = 3, n_extra_layers = 1,
#'                         act_cls = partial(nn$LeakyReLU, negative_slope = 0.2))
#'
#' }
#'
#' @export
basic_critic <- function(in_size, n_channels,
                         ...) {

  args <- list(
    in_size = in_size,
    n_channels = as.integer(n_channels),
    ...
  )

  strings = c('in_size',
              'in_sz',
              'out_size',
              'n_channels',
              'n_features',
              'n_extra_layers',
              'norm_type',
              'ks',
              'stride',
              'ndim',
              'dilation',
              'groups')

  for (i in 1:length(strings)) {
    if(!is.null(args[[strings[i]]])) {
      args[[strings[i]]] = as.integer(args[[strings[i]]])
    }
  }

  do.call(vision$gan$basic_critic, args)

}



#' @title Wgan
#'
#' @description Create a WGAN from `data`, `generator` and `critic`.
#'
#' @param dls dataloader
#' @param generator generator
#' @param critic critic
#' @param switcher switcher
#' @param clip clip value
#' @param switch_eval switch evaluation
#' @param gen_first generator first
#' @param show_img show image or not
#' @param cbs callbacks
#' @param metrics metrics
#' @param opt_func optimization function
#' @param lr learning rate
#' @param splitter splitter
#' @param path path
#' @param model_dir model directory
#' @param wd weight decay
#' @param wd_bn_bias weight decay bn bias
#' @param train_bn It controls if BatchNorm layers are trained even when they are supposed to be frozen according to the splitter.
#' @param moms momentums
#' @return None
#'
#' @examples
#' \dontrun{
#'
#' learn = GANLearner_wgan(dls, generator, critic, opt_func = partial(Adam(), mom=0.))
#'
#' }
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
#' @param object model
#' @param ... additonal parameters to pass
#' @return train history
#'
#'
#' @examples
#' \dontrun{
#'
#' learn %>% fit(1, 2e-4, wd = 0)
#'
#' }
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



#' @title GAN Module
#'
#' @description Wrapper around a `generator` and a `critic` to create a GAN.
#'
#'
#' @param generator generator
#' @param critic critic
#' @param gen_mode generator mode or not
#' @return None
#' @export
GANModule <- function(generator = NULL, critic = NULL, gen_mode = FALSE) {

  args <- list(
    generator = generator,
    critic = critic,
    gen_mode = gen_mode
  )

  do.call(vision$gan$GANModule, args)

}


#' @title GAN Discriminative LR
#'
#' @description `Callback` that handles multiplying the learning rate by `mult_lr` for the critic.
#'
#'
#' @param mult_lr mult learning rate
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
#' @return block
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


#' @title Add Channels
#'
#' @description Add `n_dim` channels at the end of the input.
#'
#'
#' @param n_dim number of dimensions
#'
#' @export
AddChannels <- function(n_dim) {

  vision$gan$AddChannels(
    n_dim = as.integer(n_dim)
  )

}




#' @title Dense Res Block
#'
#' @description Resnet block of `nf` features. `conv_kwargs` are passed to `conv_layer`.
#'
#'
#' @param nf number of features
#' @param norm_type normalization type
#' @param ks kernel size
#' @param stride stride
#' @param padding padding
#' @param bias bias
#' @param ndim number of dimensions
#' @param bn_1st batch normalization 1st
#' @param act_cls activation
#' @param transpose transpose
#' @param init initizalier
#' @param xtra xtra
#' @param bias_std bias standard deviation
#' @param dilation dilation number
#' @param groups groups number
#' @param padding_mode padding mode
#' @return block
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

#' @title Gan critic
#'
#' @description Critic to train a `GAN`.
#'
#'
#' @param n_channels number of channels
#' @param nf number of features
#' @param n_blocks number of blocks
#' @param p probability
#' @return GAN object
#' @export
gan_critic <- function(n_channels = 3, nf = 128, n_blocks = 3, p = 0.15) {

  vision$gan$gan_critic(
    n_channels = as.integer(n_channels),
    nf = as.integer(nf),
    n_blocks = as.integer(n_blocks),
    p = p
  )

}


#' @title GAN Loss
#'
#' @description Wrapper around `crit_loss_func` and `gen_loss_func`
#'
#'
#' @param gen_loss_func generator loss funcion
#' @param crit_loss_func discriminator loss function
#' @param gan_model GAN model
#' @return None
#' @export
GANLoss <- function(gen_loss_func, crit_loss_func, gan_model) {

  vision$gan$GANLoss(
    gen_loss_func = gen_loss_func,
    crit_loss_func = crit_loss_func,
    gan_model = gan_model
  )

}



#' @title Accuracy threshold expand
#'
#' @description Compute accuracy after expanding `y_true` to the size of `y_pred`.
#'
#'
#' @param y_pred predictions
#' @param y_true actuals
#' @param thresh threshold point
#' @param sigmoid sigmoid function
#' @return None
#' @export
accuracy_thresh_expand <- function(y_pred, y_true, thresh = 0.5, sigmoid = TRUE) {

  vision$gan$accuracy_thresh_expand(
    y_pred = y_pred,
    y_true = y_true,
    thresh = thresh,
    sigmoid = sigmoid
  )

}

#' @title Set freeze model
#'
#'
#' @param m parameters
#' @param rg rg
#' @return None
#' @export
set_freeze_model <- function(m, rg) {

  vision$gan$set_freeze_model(
    m = m,
    rg = rg
  )

}

#' @title GAN Trainer
#'
#' @description Handles GAN Training.
#'
#'
#' @param switch_eval switch evaluation
#' @param clip clip value
#' @param beta beta parameter
#' @param gen_first generator first
#' @param show_img show image or not
#' @return None
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


#' @title Fixed GAN Switcher
#'
#' @description Switcher to do `n_crit` iterations of the critic then `n_gen` iterations of the generator.
#'
#'
#' @param n_crit number of discriminator
#' @param n_gen number of generator
#' @return None
#' @export
FixedGANSwitcher <- function(n_crit = 1, n_gen = 1) {

  vision$gan$FixedGANSwitcher(
    n_crit = as.integer(n_crit),
    n_gen = as.integer(n_gen)
  )

}

#' @title Adaptive GAN Switcher
#'
#' @description Switcher that goes back to generator/critic when the loss goes below `gen_thresh`/`crit_thresh`.
#'
#'
#' @param gen_thresh generator threshold
#' @param critic_thresh discriminator threshold
#' @return None
#' @export
AdaptiveGANSwitcher <- function(gen_thresh = NULL, critic_thresh = NULL) {

  vision$gan$AdaptiveGANSwitcher(
    gen_thresh = gen_thresh,
    critic_thresh = critic_thresh
  )

}


#' @title Invisible Tensor
#'
#' @param x tensor
#' @return None
#' @export
InvisibleTensor <- function(x) {

  vision$gan$InvisibleTensor(
    x = x
  )

}

#' @title GAN loss from function
#'
#' @description Define loss functions for a GAN from `loss_gen` and `loss_crit`.
#'
#'
#' @param loss_gen generator loss
#' @param loss_crit discriminator loss
#' @param weights_gen weight generator
#' @return None
#' @export
gan_loss_from_func <- function(loss_gen, loss_crit, weights_gen = NULL) {

  vision$gan$gan_loss_from_func(
    loss_gen = loss_gen,
    loss_crit = loss_crit,
    weights_gen = weights_gen
  )

}


#' @title GAN Learner from learners
#'
#' @description Create a GAN from `learn_gen` and `learn_crit`.
#'
#' @param gen_learn generator learner
#' @param crit_learn discriminator learner
#' @param switcher switcher
#' @param weights_gen weights generator
#' @param gen_first generator first
#' @param switch_eval switch evaluation
#' @param show_img show image or not
#' @param clip clip value
#' @param cbs Cbs is one or a list of Callbacks to pass to the Learner.
#' @param metrics It is an optional list of metrics, that can be either functions or Metrics.
#' @param loss_func loss function
#' @param opt_func The function used to create the optimizer
#' @param lr learning rate
#' @param splitter It is a function that takes self.model and returns a list of parameter groups (or just one parameter group if there are no different parameter groups).
#' @param path The folder where to work
#' @param model_dir Path and model_dir are used to save and/or load models.
#' @param wd It is the default weight decay used when training the model.
#' @param wd_bn_bias It controls if weight decay is applied to BatchNorm layers and bias.
#' @param train_bn It controls if BatchNorm layers are trained even when they are supposed to be frozen according to the splitter.
#' @param moms The default momentums used in Learner$fit_one_cycle.
#' @return None
#' @export
GANLearner_from_learners <- function(gen_learn, crit_learn, switcher = NULL, weights_gen = NULL,
                          gen_first = FALSE, switch_eval = TRUE, show_img = TRUE,
                          clip = NULL, cbs = NULL, metrics = NULL, loss_func = NULL,
                          opt_func = Adam(), lr = 0.001, splitter = trainable_params(),
                          path = NULL, model_dir = "models", wd = NULL,
                          wd_bn_bias = FALSE, train_bn = TRUE, moms = list(0.95, 0.85, 0.95)) {

  args <- list(
    gen_learn = gen_learn,
    crit_learn = crit_learn,
    switcher = switcher,
    weights_gen = weights_gen,
    gen_first = gen_first,
    switch_eval = switch_eval,
    show_img = show_img,
    clip = clip,
    cbs = cbs,
    metrics = metrics,
    loss_func = loss_func,
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

  do.call(vision$gan$GANLearner$from_learners, args)

}




#' @title Learner
#'
#'
#' @param ... parameters to pass
#' @return None
#'
#' @examples
#' \dontrun{
#'
#' model = LitModel()
#'
#' data = Data_Loaders(model$train_dataloader(), model$val_dataloader())$cuda()
#'
#' learn = Learner(data, model, loss_func = F$cross_entropy, opt_func = Adam,
#'                 metrics = accuracy)
#'
#' }
#'
#' @export
Learner = function(...) {
  args = list(...)

  do.call(fastai2$vision$all$Learner, args)
}



#' @title Get_c
#'
#'
#' @param dls dataloader object
#' @return number of layers
#'
#' @examples
#' \dontrun{
#'
#' get_c(dls)
#'
#' }
#'
#' @export
get_c <- function(dls) {

  vision$all$get_c(
    dls = dls
  )

}








