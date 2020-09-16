
#' @title MSELossFlat
#'
#' @description Flattens input and output, same as nn$MSELoss
#'
#'
#' @export
MSELossFlat = function(...) {
  args = list(...)

  do.call(vision$all$MSELossFlat, args)
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















