#' @title Error rate
#'
#' @description 1 - `accuracy`
#'
#' @param inp The predictions of the model
#' @param targ The corresponding labels
#' @param axis Axis
#' @return tensor
#'
#' @examples
#'
#' \dontrun{
#'
#' learn = cnn_learner(dls, resnet34(), metrics = error_rate)
#'
#'
#' }
#'
#'
#' @export
error_rate <- function(inp, targ, axis = -1) {

  if(missing(inp) & missing(targ)) {
    vision()$all$error_rate
  } else {
    args <- list(
      inp = inp,
      targ = targ,
      axis = as.integer(axis)
    )

    do.call(vision()$all$error_rate, args)
  }

}

attr(error_rate ,"py_function_name") <- "error_rate"
