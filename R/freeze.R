#' Freeze a model
#'
#' @param object A model
#' @param ... Additional parameters
#'
#' @return None
#' @export
#'
#' @examples
#' \dontrun{
#' learnR %>% freeze()
#' }
freeze <- function(object, ...){

  object$freeze(...)
  message("The model has been frozen")

}

#' Unfreeze a model
#'
#' @param object A model
#' @param ... Additional parameters
#'
#' @return None
#' @export
#'
#' @examples
#'
#' \dontrun{
#' learnR %>% unfreeze()
#' }
unfreeze <- function(object, ...){

  object$unfreeze(...)
  message("The model has been unfrozen")

}




