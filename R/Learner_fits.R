



#' @title Fit_flat_lin
#'
#' @description Fit `self.model` for `n_epoch` at flat `start_lr`
#' before `curve_type` annealing to `end_lr` with weight decay of `wd` and
#' callbacks `cbs`.
#'
#' @param object model / learner
#' @param n_epochs number of epochs
#' @param n_epochs_decay number of epochs with decay
#' @param start_lr Desired starting learning rate, used for beginning pct of training.
#' @param end_lr  Desired end learning rate, training will conclude at this learning rate.
#' @param curve_type Curve type for learning rate annealing. Options are 'linear', 'cosine', and 'exponential'.
#'
#' @param wd weight decay
#' @param cbs callbacks
#' @param reset_opt reset optimizer
#' @return None
#' @export
fit_flat_lin <- function(object, n_epochs = 100, n_epochs_decay = 100,
                         start_lr = NULL, end_lr = 0, curve_type = "linear",
                         wd = NULL, cbs = NULL, reset_opt = FALSE) {

  args <- list(
    n_epochs = as.integer(n_epochs),
    n_epochs_decay = as.integer(n_epochs_decay),
    start_lr = start_lr,
    end_lr = end_lr,
    curve_type = curve_type,
    wd = wd,
    cbs = cbs,
    reset_opt = reset_opt
  )

  if(is.null(args$start_lr))
    args$start_lr <- NULL

  if(is.null(args$wd))
    args$wd <- NULL

  if(is.null(args$cbs))
    args$cbs <- NULL

  do.call(object$fit_flat_lin, args)
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


#' @title Fit_flat_cos
#'
#' @param object learner/model
#' @param n_epoch number of epochs
#' @param lr learning rate
#' @param div_final divide final value
#' @param pct_start start percentage
#' @param wd weight decay
#' @param cbs callbacks
#' @param reset_opt reset optimizer
#' @return None
#'
#'
#' @export
fit_flat_cos = function(object, n_epoch, lr=NULL, div_final=100000.0,
                        pct_start=0.75, wd=NULL, cbs=NULL, reset_opt=FALSE) {

  args = list(
    n_epoch = as.integer(n_epoch),
    lr = lr,
    div_final = div_final,
    pct_start = pct_start,
    wd = wd,
    cbs = cbs,
    reset_opt = reset_opt
  )

  if(is.null(args$lr))
    args$lr <- NULL

  if(is.null(args$wd))
    args$wd <- NULL

  if(is.null(args$cbs))
    args$cbs <- NULL

  do.call(object$fit_flat_cos, args)

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




#' @title Fit_sgdr
#'
#' @param object learner/model
#' @param n_cycles number of cycles
#' @param lr_max maximum learning rate
#' @param cycle_mult cycle mult
#' @param cycle_len length of cycle
#' @param wd weight decay
#' @param cbs callbacks
#' @param reset_opt reset optimizer
#' @return None
#'
#'
#' @export
fit_sgdr = function(object, n_cycles, cycle_len, lr_max=NULL,
                        cycle_mult=2, cbs=NULL, reset_opt=FALSE, wd=NULL) {

  args = list(
    n_cycles = as.integer(n_cycles),
    cycle_len = as.integer(cycle_len),
    lr_max = lr_max,
    cycle_mult = as.integer(cycle_mult),
    cbs = cbs,
    reset_opt = reset_opt,
    wd = wd
  )

  if(is.null(args$lr_max))
    args$lr_max <- NULL

  if(is.null(args$wd))
    args$wd <- NULL

  if(is.null(args$cbs))
    args$cbs <- NULL

  do.call(object$fit_sgdr, args)

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



#' @title Fine_tune
#'
#' @description Fine tune with `freeze` for `freeze_epochs` then
#' with `unfreeze` from `epochs` using discriminative LR
#'
#' @param object learner/model
#' @param epochs epoch number
#' @param base_lr base learning rate
#' @param freeze_epochs freeze epochs number
#' @param lr_mult learning rate multiply
#' @param pct_start start percentage
#' @param div divide
#' @param ... additional arguments
#' @return None
#' @export
fine_tune <- function(object, epochs, base_lr = 0.002, freeze_epochs = 1,
                      lr_mult = 100, pct_start = 0.3, div = 5.0,
                      ...) {

  args <- list(
    epochs = as.integer(epochs),
    base_lr = base_lr,
    freeze_epochs = as.integer(freeze_epochs),
    lr_mult = as.integer(lr_mult),
    pct_start = pct_start,
    div = div,
    ...
  )

  do.call(object$fine_tune, args)

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



