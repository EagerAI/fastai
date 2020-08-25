#' @title DataBlock
#'
#' @description Generic container to quickly build `Datasets` and `DataLoaders`
#'
#'
#' @param blocks blocks
#' @param dl_type dl_type
#' @param getters getters
#' @param n_inp n_inp
#' @param item_tfms item_tfms
#' @param batch_tfms batch_tfms
#'
#' @export
DataBlock <- function(blocks = NULL, dl_type = NULL, getters = NULL,
                      n_inp = NULL, item_tfms = NULL, batch_tfms = NULL,
                      ...) {

  args <- list(
    blocks = blocks,
    dl_type = dl_type,
    getters = getters,
    n_inp = n_inp,
    item_tfms = item_tfms,
    batch_tfms = batch_tfms,
    ...
  )

  do.call(vision$gan$DataBlock, args)

}



#' @title TransformBlock
#'
#' @description A basic wrapper that links defaults transforms for the data block API
#'
#'
#' @param type_tfms type_tfms
#' @param item_tfms item_tfms
#' @param batch_tfms batch_tfms
#' @param dl_type dl_type
#' @param dls_kwargs dls_kwargs
#'
#' @export
TransformBlock <- function(type_tfms = NULL, item_tfms = NULL,
                           batch_tfms = NULL, dl_type = NULL,
                           dls_kwargs = NULL) {



  if(missing(type_tfms) & missing(item_tfms) & missing(batch_tfms) & missing(dl_type) & missing(dls_kwargs)) {
    invisible(vision$gan$TransformBlock)
  } else {
    args <- list(
      type_tfms = type_tfms,
      item_tfms = item_tfms,
      batch_tfms = batch_tfms,
      dl_type = dl_type,
      dls_kwargs = dls_kwargs
    )
    do.call(vision$gan$TransformBlock, args)
  }


}


#' @title ImageBlock
#'
#' @description A `TransformBlock` for images of `cls`
#'
#'
#' @export
ImageBlock <- function() {

  invisible(vision$gan$ImageBlock)

}

#' @title Generate_noise
#'
#'
#' @param fn fn
#' @param size size
#'
#' @export
generate_noise <- function(fn, size = 100) {



  if(missing(fn)) {
    invisible(vision$gan$generate_noise)
  } else {
    args <- list(
      fn = fn,
      size = as.integer(size)
    )
    do.call(vision$gan$generate_noise,args)
  }


}

#' @title IndexSplitter
#'
#' @description Split `items` so that `val_idx` are in the validation set and the others in the training set
#'
#'
#' @param valid_idx valid_idx
#'
#' @export
IndexSplitter <- function(valid_idx) {

  if(missing(valid_idx)) {
    invisible(vision$gan$IndexSplitter)
  } else {
    args <- list(
      valid_idx = valid_idx
    )
    do.call(vision$gan$IndexSplitter,args)
  }


}


#' @title dataloaders
#'
#' @description Create a `DataLoaders` object from `source`
#'
#' @details
#'
#' @param source source
#' @param path path
#' @param verbose verbose
#' @param bs bs
#' @param shuffle shuffle
#' @param num_workers num_workers
#' @param do_setup do_setup
#' @param pin_memory pin_memory
#' @param timeout timeout
#' @param batch_size batch_size
#' @param drop_last drop_last
#' @param indexed indexed
#' @param n n
#' @param device device
#'
#' @export
dataloaders <- function(object, source, path = ".", verbose = FALSE, bs = 64,
                        shuffle = FALSE, num_workers = NULL, do_setup = TRUE,
                        pin_memory = FALSE, timeout = 0, batch_size = NULL,
                        drop_last = FALSE, indexed = NULL, n = NULL, device = NULL) {

  args <- list(
    source = source,
    path = path,
    verbose = verbose,
    bs = as.integer(bs),
    shuffle = shuffle,
    num_workers = num_workers,
    do_setup = do_setup,
    pin_memory = pin_memory,
    timeout = as.integer(timeout),
    batch_size = batch_size,
    drop_last = drop_last,
    indexed = indexed,
    n = n,
    device = device
  )

  if(!is.null(batch_size)) {
    args$batch_size <- as.integer(args$batch_size)
  }

  do.call(object$dataloaders,args)
}





