


#' @title Dataloader
#'
#'
#' @param ... parameters to pass
#'
#'
#' @export
Data_Loaders = function(...) {

  args = list(...)
  args = unlist(args)

  if(is.list(args[[1]]) & length(args[[1]])==2) {
    args = append(args[[1]][[1]], args[[1]][[2]], args[2:length(args)])
    do.call(fastai2$vision$all$DataLoaders, args)
  } else if (length(args)==2) {
    args = unlist(args)
    fastai2$vision$all$DataLoaders(args[[1]], args[[2]])
  } else {
    do.call(fastai2$vision$all$DataLoaders, args)
  }

}




