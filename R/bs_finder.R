


#' @title Bs_find
#'
#' @description Launch a mock training to find a good batch size to minimaze training time.
#'
#' @details However, it may not be a good batch size to minimize the validation loss. A good batch size is where the Simple Noise Scale converge ignoring the small growing trend with the number of iterations if exists. The optimal batch size is about an order the magnitud
#' where Simple Noise scale converge. Typically, the optimial batch size in image classification problems will be 2-3 times lower where
#'
#' @param object model/learner
#' @param lr learning rate
#' @param num_it number of iterations
#' @param n_batch number of batches
#' @param simulate_multi_gpus simulate on multi gpus or not
#' @param show_plot show plot or not
#'
#' @export
bs_find <- function(object, lr, num_it = NULL, n_batch = 5, simulate_multi_gpus = TRUE, show_plot = TRUE) {

  args <- list(
    lr = lr,
    num_it = num_it,
    n_batch = as.integer(n_batch),
    simulate_multi_gpus = simulate_multi_gpus,
    show_plot = show_plot
  )

  do.call(object$bs_find, args)

  invisible(object$recorder$bs_find_stats)
}

#' @title Plot_bs_find
#'
#'
#' @param object model
#' @param ... additional arguments
#' @param dpi dots per inch
#' @return None
#'
#' @export
plot_bs_find <- function(object, ..., dpi = 250) {

  fastai2$vision$all$plt$close()
  object$recorder$plot_bs_find(
    ...
  )

  tmp_d = gsub(tempdir(), replacement = '/', pattern = '\\', fixed = TRUE)
  fastai2$tabular$all$plt$savefig(paste(tmp_d, 'test.png', sep = '/'), dpi = as.integer(dpi))

  img <- png::readPNG(paste(tmp_d, 'test.png', sep = '/'))
  if(!is_rmarkdown()) {
    try(dev.off(),TRUE)
  }
  grid::grid.raster(img)

}




