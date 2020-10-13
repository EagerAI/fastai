#' @title Cnn_learner
#'
#' @description Build a convnet style learner from `dls` and `arch`
#'
#'
#' @param dls data loader object
#' @param arch a model architecture
#' @param loss_func loss function
#' @param pretrained pre-trained or not
#' @param cut cut
#' @param splitter It is a function that takes self.model and returns a list of parameter groups (or just one parameter group if there are no different parameter groups).
#' @param y_range y_range
#' @param config configuration
#' @param n_out the number of out
#' @param normalize normalize
#' @param opt_func The function used to create the optimizer
#' @param lr learning rate
#' @param cbs Cbs is one or a list of Callbacks to pass to the Learner.
#' @param metrics It is an optional list of metrics, that can be either functions or Metrics.
#' @param path The folder where to work
#' @param model_dir Path and model_dir are used to save and/or load models.
#' @param wd It is the default weight decay used when training the model.
#' @param wd_bn_bias It controls if weight decay is applied to BatchNorm layers and bias.
#' @param train_bn It controls if BatchNorm layers are trained even when they are supposed to be frozen according to the splitter.
#' @param moms The default momentums used in Learner.fit_one_cycle.
#' @return learner object
#'
#' @examples
#'
#' \dontrun{
#'
#' URLs_MNIST_SAMPLE()
#' # transformations
#' tfms = aug_transforms(do_flip = FALSE)
#' path = 'mnist_sample'
#' bs = 20
#'
#' #load into memory
#' data = ImageDataLoaders_from_folder(path, batch_tfms = tfms, size = 26, bs = bs)
#'
#'
#' learn = cnn_learner(data, resnet18(), metrics = accuracy, path = getwd())
#'
#' }
#'
#' @export
cnn_learner <- function(dls, arch, loss_func = NULL, pretrained = TRUE, cut = NULL,
                        splitter = NULL, y_range = NULL, config = NULL, n_out = NULL,
                        normalize = TRUE, opt_func = Adam(), lr = 0.001, cbs = NULL,
                        metrics = NULL, path = NULL, model_dir = "models", wd = NULL,
                        wd_bn_bias = FALSE, train_bn = TRUE, moms = list(0.95, 0.85, 0.95)) {

 args <- list(
    dls = dls,
    arch = arch,
    loss_func = loss_func,
    pretrained = pretrained,
    cut = cut,
    splitter = splitter,
    y_range = y_range,
    config = config,
    n_out = n_out,
    normalize = normalize,
    opt_func = opt_func,
    lr = lr,
    cbs = cbs,
    metrics = metrics,
    path = path,
    model_dir = model_dir,
    wd = wd,
    wd_bn_bias = wd_bn_bias,
    train_bn = train_bn,
    moms = moms
  )


 do.call(vision$all$cnn_learner,args)

}

#' @title Fit
#' @description Fit the model on this learner with `lr` learning rate,
#' `wd` weight decay for `epochs` with `callbacks` as cbs argument.
#' @param object a learner object
#' @param ... parameters to pass
#' @importFrom generics fit
#' @return train history
#' @export
fit.fastai.learner.Learner <- function(object, ...) {

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

#' @title Cnn config
#'
#' @description Convenience function to easily create a config for `create_cnn_model`
#'
#'
#' @param cut cut
#' @param pretrained pretrained
#' @param n_in input shape
#' @param init init
#' @param custom_head custom_head
#' @param concat_pool concat_pool
#' @param lin_ftrs lin_ftrs
#' @param ps ps
#' @param bn_final bn_final
#' @param lin_first lin_first
#' @param y_range y_range
#' @return None
#' @export
cnn_config <- function(cut = NULL, pretrained = TRUE, n_in = 3,
                       init = nn$init$kaiming_normal_, custom_head = NULL,
                       concat_pool = TRUE, lin_ftrs = NULL, ps = 0.5,
                       bn_final = FALSE, lin_first = FALSE, y_range = NULL) {

  args <- list(
    cut = cut,
    pretrained = pretrained,
    n_in = as.integer(n_in),
    init = init,
    custom_head = custom_head,
    concat_pool = concat_pool,
    lin_ftrs = lin_ftrs,
    ps = ps,
    bn_final = bn_final,
    lin_first = lin_first,
    y_range = y_range
  )

  do.call(vision$all$cnn_config, args)

}


#' @title Create_cnn_model
#'
#' @description Create custom convnet architecture using `arch`, `n_in` and `n_out`
#'
#' @param arch a model architecture
#' @param n_out number of out
#' @param cut cut
#' @param pretrained pretrained model or not
#' @param n_in input shape
#' @param init initializer
#' @param custom_head custom head
#' @param concat_pool concatenate pooling
#' @param lin_ftrs lin_ftrs
#' @param ps ps
#' @param bn_final bn_final
#' @param lin_first lin_first
#' @param y_range y_range
#' @return None
#' @export
create_cnn_model <- function(arch, n_out, cut = NULL, pretrained = TRUE,
                             n_in = 3, init = nn$init$kaiming_normal_,
                             custom_head = NULL, concat_pool = TRUE,
                             lin_ftrs = NULL, ps = 0.5, bn_final = FALSE,
                             lin_first = FALSE, y_range = NULL) {

  args <- list(
    arch = arch,
    n_out = n_out,
    cut = cut,
    pretrained = pretrained,
    n_in = as.integer(n_in),
    init = init,
    custom_head = custom_head,
    concat_pool = concat_pool,
    lin_ftrs = lin_ftrs,
    ps = ps,
    bn_final = bn_final,
    lin_first = lin_first,
    y_range = y_range
  )

  do.call(vision$all$create_cnn_model, args)

}




