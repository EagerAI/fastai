


#' @title TabularTS
#'
#' @description A `DataFrame` wrapper that knows which cols are x/y, and returns rows in `__getitem__`
#'
#'
#' @param df A DataFrame of your data
#' @param procs list of preprocess functions
#' @param x_names predictors names
#' @param y_names the names of the dependent variables
#' @param block_y the TransformBlock to use for the target
#' @param splits How to split your data
#' @param do_setup A parameter for if Tabular will run the data through the procs upon initialization
#' @param device device name
#' @param inplace If True, Tabular will not keep a separate copy of your original DataFrame in memory
#' @return None
#' @export
TabularTS <- function(df, procs = NULL, x_names = NULL, y_names = NULL,
                      block_y = NULL, splits = NULL, do_setup = TRUE,
                      device = NULL, inplace = FALSE) {

  tms()$tabular$TabularTS(
    df = df,
    procs = procs,
    x_names = x_names,
    y_names = y_names,
    block_y = block_y,
    splits = splits,
    do_setup = do_setup,
    device = device,
    inplace = inplace
  )

}


#' @title TSDataTable
#'
#' @description A `DataFrame` wrapper that knows which cols are x/y, and returns rows in `__getitem__`
#'
#'
#' @param df A DataFrame of your data
#' @param procs list of preprocess functions
#' @param x_names predictors names
#' @param y_names the names of the dependent variables
#' @param block_y the TransformBlock to use for the target
#' @param splits How to split your data
#' @param do_setup A parameter for if Tabular will run the data through the procs upon initialization
#' @param device device name
#' @param inplace If True, Tabular will not keep a separate copy of your original DataFrame in memory
#' @return None
#' @export
TSDataTable <- function(df, procs = NULL, x_names = NULL, y_names = NULL,
                     block_y = NULL, splits = NULL, do_setup = TRUE,
                     device = NULL, inplace = FALSE) {

  args = list(
    df = df,
    procs = procs,
    x_names = x_names,
    y_names = y_names,
    block_y = block_y,
    splits = splits,
    do_setup = do_setup,
    device = device,
    inplace = inplace
  )

  if(!is.null(splits))
    args$splits = list(as.integer(splits[[1]]-1),as.integer(splits[[2]]-1))

  do.call(tms()$tabular$TSPandas, args)

}


#' @title NormalizeTS
#'
#' @description Normalize the x variables.
#'
#'
#' @param enc encoder
#' @param dec decoder
#' @param split_idx split by index
#' @param order order
#' @return None
#' @export
NormalizeTS <- function(enc = NULL, dec = NULL, split_idx = NULL, order = NULL) {

  args = list(
    enc = enc,
    dec = dec,
    split_idx = split_idx,
    order = order
  )

  if(!is.null(args[['split_idx']])) {
    args[['split_idx']] = as.integer(args[['split_idx']])
  }

  do.call(tms()$tabular$NormalizeTS, args)

}


#' @title ReadTSBatch
#'
#' @description A transform that always take lists as items
#'
#'
#' @param to output from TSDataTable function
#' @return None
#' @export
ReadTSBatch <- function(to) {

  tms()$tabular$ReadTSBatch(
    to = to
  )

}



#' @title TabularTSDataloader
#'
#' @description Transformed `DataLoader`
#'
#'
#' @param dataset data set
#' @param bs batch size
#' @param shuffle shuffle or not
#' @param after_batch after batch
#' @param num_workers the number of workers
#' @param verbose verbose
#' @param do_setup A parameter for if Tabular will run the data through the procs upon initialization
#' @param pin_memory pin memory or not
#' @param timeout timeout
#' @param batch_size batch size
#' @param drop_last drop last
#' @param indexed indexed
#' @param n n
#' @param device device name
#' @return None
#' @export
TabularTSDataloader <- function(dataset, bs = 16, shuffle = FALSE, after_batch = NULL,
                                num_workers = 0, verbose = FALSE, do_setup = TRUE,
                                pin_memory = FALSE, timeout = 0, batch_size = NULL,
                                drop_last = FALSE, indexed = NULL, n = NULL, device = NULL) {

  args <- list(
    dataset = dataset,
    bs = as.integer(bs),
    shuffle = shuffle,
    after_batch = after_batch,
    num_workers = as.integer(num_workers),
    verbose = verbose,
    do_setup = do_setup,
    pin_memory = pin_memory,
    timeout = as.integer(timeout),
    batch_size = batch_size,
    drop_last = drop_last,
    indexed = indexed,
    n = n,
    device = device
  )

  if(!is.null(args[['batch_size']]))
    args[['batch_size']] = as.integer(args[['batch_size']])

  do.call(tms()$tabular$TabularTSDataloader, args)

}


