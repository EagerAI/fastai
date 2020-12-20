#' Freeze a model
#'
#' @param object A model 
#'
#' @return Nothing
#' @export
#'
#' @examples
#' \donrun{
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
#' \donrun{
#' learnR %>% unfreeze()
#' }
unfreeze <- function(object){
  
  args <- list()
  
  do.call(object$unfreeze, args)
  message("The model has been unfreezed")
  
}
  
    

    
