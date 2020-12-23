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
  
  args <- list(...)
  
  do.call(object$freeze, args)
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
  
  args <- list(...)
  
  do.call(object$unfreeze, args)
  message("The model has been unfrozen")
  
}
  
    

    
