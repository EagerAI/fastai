#' @title Error_rate
#'
#' @description 1 - `accuracy`
#'
#' @param inp The predictions of the model
#' @param targ The corresponding labels
#' @param axis Axis
#'
#' @export
error_rate <- function(inp, targ, axis = -1) {

  if(!missing(inp) && !missing(targ)) {
    args <- list(
      inp = inp,
      targ = targ,
      axis = as.integer(axis)
    )

    do.call(vision$all$error_rate, args)
  } else {
    vision$all$error_rate
  }

}

attr(error_rate ,"py_function_name") <- "error_rate"
