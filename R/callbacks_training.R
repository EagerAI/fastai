


#' @title ShortEpochCallback
#'
#' @description Fit just `pct` of an epoch, then stop
#'
#'
#' @param pct percentage
#' @param short_valid short_valid or not
#' @return None
#'
#' @export
ShortEpochCallback <- function(pct = 0.01, short_valid = TRUE) {

  fastai2$callback$training$ShortEpochCallback(
    pct = pct,
    short_valid = short_valid
  )

}


#' @title GradientAccumulation
#'
#' @description Accumulate gradients before updating weights
#'
#'
#' @param n_acc number of acc
#' @return None
#' @export
GradientAccumulation <- function(n_acc = 32) {

  fastai2$callback$training$GradientAccumulation(
    n_acc = as.integer(n_acc)
  )

}


