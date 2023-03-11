


#' @title Data Loaders
#' @param ... parameters to pass
#'
#' @return loader object
#'
#' @examples
#'
#' \dontrun{
#'
#' data = Data_Loaders(train_loader, test_loader)
#'
#' learn = Learner(data, Net(), loss_func = F$nll_loss,
#'                 opt_func = Adam(), metrics = accuracy, cbs = CudaCallback())
#'
#' learn %>% fit_one_cycle(1, 1e-2)
#'
#' }
#'
#' @export
Data_Loaders = function(...) {

  args = list(...)

  # put into GPU
  if(torch()$cuda$is_available() & is.null(args$device)) {
    args = append(args, list(device = 'cuda'))
  }

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




