#' @title Cnn_learner
#'
#' @description Build a convnet style learner from `dls` and `arch`
#'
#'
#' @param dls dls
#' @param arch arch
#' @param loss_func loss_func
#' @param pretrained pretrained
#' @param cut cut
#' @param splitter splitter
#' @param y_range y_range
#' @param config config
#' @param n_out n_out
#' @param normalize normalize
#' @param opt_func opt_func
#' @param lr lr
#' @param cbs cbs
#' @param metrics metrics
#' @param path path
#' @param model_dir model_dir
#' @param wd wd
#' @param wd_bn_bias wd_bn_bias
#' @param train_bn train_bn
#' @param moms moms
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
#' @description Fit the model on this learner with `lr` learning rate, `wd` weight decay for `epochs` with `callbacks`.
#'
#' @param epochs epochs
#' @param lr lr
#' @param wd wd
#' @param callbacks callbacks
#'
#' @export
fit.fastai2.learner.Learner <- function(object, n_epoch, lr = 1e-2, wd = NULL, callbacks = NULL) {

  args <- list(
    n_epoch = as.integer(n_epoch),
    lr = lr,
    wd = wd,
    callbacks = callbacks
  )

  # fit the model
  do.call(object$fit, args)

}



