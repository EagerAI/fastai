#' Freeze a model
#'
#' @param object A model 
#'
#' @return Nothing
#' @export
#'
#' @examples
#' \dontrun{
#' learnR %>% freeze()
#' }
freeze <- function(object){
  
  args <- list()
  
  do.call(object$freeze, args)
  message("The model has been freezed")
  
}

#' Unfreeze a model
#'
#' @param object A model 
#'
#' @return Nothing
#' @export
#'
#' @examples
#' 
#' \dontrun{
#' learnR %>% unfreeze()
#' }
unfreeze <- function(object){
  
  args <- list()
  
  do.call(object$unfreeze, args)
  message("The model has been unfreezed")
  
}
  
    

    
