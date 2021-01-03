
##################################################  Faster RSNN train dataloader

#' @title Faster RSNN train dataloader
#'
#' @description A `DataLoader` with a custom `collate_fn` that batches items as required for training the model.
#'
#' @param dataset Possibly a `Dataset` object, but more generally, any `Sequence` that returns records.
#' @param batch_tfms Transforms to be applied at the batch level.
#' @return None
#' @param ... dataloader_kwargs: Keyword arguments that will be internally passed to a Pytorch `DataLoader`. The parameter `collate_fn` is already defined internally and cannot be passed here.
#' @export
faster_rcnn_train_dl <- function(dataset, batch_tfms = NULL, ...) {

  args <- list(
    dataset = dataset,
    batch_tfms = batch_tfms,
    ...
  )

  if(is.null(args$batch_tfms))
    args$batch_tfms <- NULL


  if(!is.null(args$batch_size)){
    args$batch_size = as.integer(args$batch_size)
  }


  if(os()=='windows' & is.null(args$num_workers)) {
    args$num_workers = 0L
  }

  if(os()=='mac' & is.null(args$num_workers)) {
    args$num_workers = 0L
  }

  if(!is.null(args$num_workers)){
    args$num_workers = as.integer(args$num_workers)
  }

  do.call(icevision()$faster_rcnn$train_dl, args)

}


#' @title Faster RSNN valid dataloader
#'
#' @description A `DataLoader` with a custom `collate_fn` that batches items as required for training the model.
#'
#' @param dataset Possibly a `Dataset` object, but more generally, any `Sequence` that returns records.
#' @param batch_tfms Transforms to be applied at the batch level.
#' @return None
#' @param ... dataloader_kwargs: Keyword arguments that will be internally passed to a Pytorch `DataLoader`. The parameter `collate_fn` is already defined internally and cannot be passed here.
#' @export
faster_rcnn_valid_dl <- function(dataset, batch_tfms = NULL, ...) {

  args <- list(
    dataset = dataset,
    batch_tfms = batch_tfms,
    ...
  )


  if(!is.null(args$batch_size)){
    args$batch_size = as.integer(args$batch_size)
  }


  if(os()=='windows' & is.null(args$num_workers)) {
    args$num_workers = 0L
  }

  if(os()=='mac' & is.null(args$num_workers)) {
    args$num_workers = 0L
  }

  if(!is.null(args$num_workers)){
    args$num_workers = as.integer(args$num_workers)
  }

  do.call(icevision()$faster_rcnn$valid_dl, args)

}

##################################################  MaskRCNN train dataloader

#' @title Efficientdet train dataloader
#'
#' @description A `DataLoader` with a custom `collate_fn` that batches items as required for training the model.
#'
#' @param dataset Possibly a `Dataset` object, but more generally, any `Sequence` that returns records.
#' @param batch_tfms Transforms to be applied at the batch level.
#' @return None
#' @param ... dataloader_kwargs: Keyword arguments that will be internally passed to a Pytorch `DataLoader`. The parameter `collate_fn` is already defined internally and cannot be passed here.
#' @export
efficientdet_train_dl <- function(dataset, batch_tfms = NULL, ...) {

  args <- list(
    dataset = dataset,
    batch_tfms = batch_tfms,
    ...
  )

  if(is.null(args$batch_tfms))
    args$batch_tfms <- NULL


  if(!is.null(args$batch_size)){
    args$batch_size = as.integer(args$batch_size)
  }


  if(os()=='windows' & is.null(args$num_workers)) {
    args$num_workers = 0L
  }

  if(os()=='mac' & is.null(args$num_workers)) {
    args$num_workers = 0L
  }

  if(!is.null(args$num_workers)){
    args$num_workers = as.integer(args$num_workers)
  }

  do.call(icevision()$efficientdet$train_dl, args)

}


#' @title Efficientdet valid dataloader
#'
#' @description A `DataLoader` with a custom `collate_fn` that batches items as required for training the model.
#'
#' @param dataset Possibly a `Dataset` object, but more generally, any `Sequence` that returns records.
#' @param batch_tfms Transforms to be applied at the batch level.
#' @return None
#' @param ... dataloader_kwargs: Keyword arguments that will be internally passed to a Pytorch `DataLoader`. The parameter `collate_fn` is already defined internally and cannot be passed here.
#' @export
efficientdet_valid_dl <- function(dataset, batch_tfms = NULL, ...) {

  args <- list(
    dataset = dataset,
    batch_tfms = batch_tfms,
    ...
  )


  if(!is.null(args$batch_size)){
    args$batch_size = as.integer(args$batch_size)
  }


  if(os()=='windows' & is.null(args$num_workers)) {
    args$num_workers = 0L
  }

  if(os()=='mac' & is.null(args$num_workers)) {
    args$num_workers = 0L
  }

  if(!is.null(args$num_workers)){
    args$num_workers = as.integer(args$num_workers)
  }

  do.call(icevision()$efficientdet$valid_dl, args)

}


##################################################  Mask_rcnn train dataloader


#' @title MaskRCNN train dataloader
#'
#' @description A `DataLoader` with a custom `collate_fn` that batches items as required for training the model.
#'
#' @param dataset Possibly a `Dataset` object, but more generally, any `Sequence` that returns records.
#' @param batch_tfms Transforms to be applied at the batch level.
#' @return None
#' @param ... dataloader_kwargs: Keyword arguments that will be internally passed to a Pytorch `DataLoader`. The parameter `collate_fn` is already defined internally and cannot be passed here.
#' @export
mask_rcnn_train_dl <- function(dataset, batch_tfms = NULL, ...) {

  args <- list(
    dataset = dataset,
    batch_tfms = batch_tfms,
    ...
  )

  if(is.null(args$batch_tfms))
    args$batch_tfms <- NULL


  if(!is.null(args$batch_size)){
    args$batch_size = as.integer(args$batch_size)
  }


  if(os()=='windows' & is.null(args$num_workers)) {
    args$num_workers = 0L
  }

  if(os()=='mac' & is.null(args$num_workers)) {
    args$num_workers = 0L
  }

  if(!is.null(args$num_workers)){
    args$num_workers = as.integer(args$num_workers)
  }

  do.call(icevision()$mask_rcnn$train_dl, args)

}


#' @title MaskRSNN valid dataloader
#'
#' @description A `DataLoader` with a custom `collate_fn` that batches items as required for training the model.
#'
#' @param dataset Possibly a `Dataset` object, but more generally, any `Sequence` that returns records.
#' @param batch_tfms Transforms to be applied at the batch level.
#' @return None
#' @param ... dataloader_kwargs: Keyword arguments that will be internally passed to a Pytorch `DataLoader`. The parameter `collate_fn` is already defined internally and cannot be passed here.
#' @export
mask_rcnn_valid_dl <- function(dataset, batch_tfms = NULL, ...) {

  args <- list(
    dataset = dataset,
    batch_tfms = batch_tfms,
    ...
  )


  if(!is.null(args$batch_size)){
    args$batch_size = as.integer(args$batch_size)
  }


  if(os()=='windows' & is.null(args$num_workers)) {
    args$num_workers = 0L
  }

  if(os()=='mac' & is.null(args$num_workers)) {
    args$num_workers = 0L
  }

  if(!is.null(args$num_workers)){
    args$num_workers = as.integer(args$num_workers)
  }

  do.call(icevision()$mask_rcnn$valid_dl, args)

}




