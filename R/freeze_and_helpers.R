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


#' Create a slice object for fine tuning
#' 
#' In transfert learning, the recommended approach to fine_tune a model is to use 
#' a discriminative learning rate, i.e. having a smaller learning rate for the 
#' earlier layers of the models, and a bigger one for the last layers. This allow 
#' to preserve the weight of the layers that recognize basic features.
#' The fit_one_cycle() method of Fastai accept a slice object as argument to 
#' the parameter lr_max, to perform fine tuning using a discriminative learning 
#' rate. This function is meant to create a slice object from R conveniently. 
#' Please refer to the page 201 of the Fastai book, or the beginner tutorial of 
#' the text tutorial on the fastai website.
#'
#' @param lr_earliest_layers Learning rate in the earliest layer
#' @param lr_last_layers Learning rate in the last layer
#'
#' @return A slice object
#' @export
#'
#' @examples
#' 
#' \dontrun{
#' sl <- create_slice(1e-6, 1e-4)
#' learnR %>% fit_one_cycle(n_epoch = 4, lr_max=sl)
#' }
create_slice <- function(lr_earliest_layer, lr_last_layer) {
  
  sl <- reticulate::import_builtins(convert = FALSE)$slice(lr_earliest_layer, lr_last_layer)
  
  return(sl)
}   

    
