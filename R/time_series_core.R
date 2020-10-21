


#' @title Maybe_unsqueeze
#'
#' @description Add empty dimension if it is a rank 1 tensor/array
#'
#'
#' @param x R array/matrix/tensor
#' @return array
#' @export
maybe_unsqueeze <- function(x) {

  tms$core$maybe_unsqueeze(
    x = x
  )

}


#' @title Show_array
#'
#' @description Show an array on `ax`.
#'
#'
#' @param array R array
#' @param ax axis
#' @param figsize figure size
#' @param title title, text
#' @param ctx ctx
#' @param tx tx
#' @return None
#' @export
show_array <- function(array, ax = NULL, figsize = NULL, title = NULL, ctx = NULL, tx = NULL) {

  tms$core$show_array(
    array = array,
    ax = ax,
    figsize = figsize,
    title = title,
    ctx = ctx,
    tx = tx
  )

}


#' @title TSeries_create
#'
#'
#' @param x tensor
#' @param ... additional parameters
#' @return tensor
#' @export
TSeries_create <- function(x, ...) {

  tms$core$TSeries$create(
    x = x,
    ...
  )

}






