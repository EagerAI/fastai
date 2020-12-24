


#' @title CollectDataCallback
#'
#' @description Collect all batches, along with pred and loss, into self.data. Mainly for testing
#'
#' @param ... arguments to pass
#' @return None
#'
#' @export
CollectDataCallback = function(...) {
  args = list(...)

  if(length(args)>0) {
    do.call(fastai2$callback$data$CollectDataCallback,args)
  } else {
    fastai2$callback$data$CollectDataCallback
  }
}


#' @title WeightedDL
#'
#' @description Transformed `DataLoader`
#'
#'
#' @param dataset dataset
#' @param bs bs
#' @param wgts weights
#' @param shuffle shuffle
#' @param num_workers number of workers
#' @param verbose verbose
#' @param do_setup do_setup
#' @param pin_memory pin_memory
#' @param timeout timeout
#' @param batch_size batch_size
#' @param drop_last drop_last
#' @param indexed indexed
#' @param n n
#' @param device device
#' @param persistent_workers persistent_workers
#' @return None
#' @export
WeightedDL <- function(dataset = NULL, bs = NULL, wgts = NULL, shuffle = FALSE,
                       num_workers = NULL, verbose = FALSE, do_setup = TRUE,
                       pin_memory = FALSE, timeout = 0, batch_size = NULL,
                       drop_last = FALSE, indexed = NULL, n = NULL,
                       device = NULL, persistent_workers = FALSE) {

  args <- list(
    dataset = dataset,
    bs = bs,
    wgts = wgts,
    shuffle = shuffle,
    num_workers = num_workers,
    verbose = verbose,
    do_setup = do_setup,
    pin_memory = pin_memory,
    timeout = as.integer(timeout),
    batch_size = batch_size,
    drop_last = drop_last,
    indexed = indexed,
    n = n,
    device = device,
    persistent_workers = persistent_workers
  )

  if(is.null(args$dataset))
    args$dataset <- NULL

  if(is.null(args$bs))
    args$bs <- NULL

  if(is.null(args$wgts))
    args$wgts <- NULL

  if(is.null(args$num_workers) & os()=="linux")
    args$num_workers<- NULL
  else if (!is.null(args$num_workers) & os()=="linux")
    args$num_workers <- as.integer(args$num_workers)
  else
    args$num_workers <- 0L

  if(is.null(args$batch_size))
    args$batch_size <- NULL

  if(is.null(args$indexed))
    args$indexed <- NULL

  if(is.null(args$n))
    args$n <- NULL

  if(is.null(args$device))
    args$device <- NULL

  do.call(fastai2$callback$data$WeightedDL, args)

}




#' @title PartialDL
#'
#' @description Select randomly partial quantity of data at each epoch
#'
#'
#' @param dataset dataset
#' @param bs bs
#' @param partial_n partial_n
#' @param shuffle shuffle
#' @param num_workers num_workers
#' @param verbose verbose
#' @param do_setup do_setup
#' @param pin_memory pin_memory
#' @param timeout timeout
#' @param batch_size batch_size
#' @param drop_last drop_last
#' @param indexed indexed
#' @param n n
#' @param device device
#' @param persistent_workers persistent_workers
#' @return None
#' @export
PartialDL <- function(dataset = NULL, bs = NULL, partial_n = NULL, shuffle = FALSE,
                      num_workers = NULL, verbose = FALSE, do_setup = TRUE,
                      pin_memory = FALSE, timeout = 0, batch_size = NULL,
                      drop_last = FALSE, indexed = NULL, n = NULL, device = NULL,
                      persistent_workers = FALSE) {

  args = list(
    dataset = dataset,
    bs = bs,
    partial_n = partial_n,
    shuffle = shuffle,
    num_workers = num_workers,
    verbose = verbose,
    do_setup = do_setup,
    pin_memory = pin_memory,
    timeout = as.integer(timeout),
    batch_size = batch_size,
    drop_last = drop_last,
    indexed = indexed,
    n = n,
    device = device,
    persistent_workers = persistent_workers
  )

  if(is.null(args$dataset))
    args$dataset <- NULL

  if(is.null(args$bs))
    args$bs <- NULL

  if(is.null(args$partial_n))
    args$partial_n <- NULL

  if(is.null(args$num_workers) & os()=="linux")
    args$num_workers<- NULL
  else if (!is.null(args$num_workers) & os()=="linux")
    args$num_workers <- as.integer(args$num_workers)
  else
    args$num_workers <- 0L

  if(is.null(args$batch_size))
    args$batch_size <- NULL

  if(is.null(args$indexed))
    args$indexed <- NULL

  if(is.null(args$n))
    args$n <- NULL

  if(is.null(args$device))
    args$device <- NULL


  do.call(fastai2$callback$data$PartialDL, args)

}










