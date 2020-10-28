



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

  do.call(object$fit_flat_lin, args)

}














