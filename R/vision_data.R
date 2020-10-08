
#' @title Get_grid
#'
#' @description Return a grid of `n` axes, `rows` by `cols`
#'
#'
#' @param n n
#' @param nrows nrows
#' @param ncols ncols
#' @param add_vert add_vert
#' @param figsize figsize
#' @param double double
#' @param title title
#' @param return_fig return_fig
#' @param imsize imsize
#' @return None
#' @export
get_grid <- function(n, nrows = NULL, ncols = NULL, add_vert = 0,
                     figsize = NULL, double = FALSE, title = NULL,
                     return_fig = FALSE, imsize = 3) {

  args = list(
    n = as.integer(n),
    nrows = nrows,
    ncols = ncols,
    add_vert = as.integer(add_vert),
    figsize = figsize,
    double = double,
    title = title,
    return_fig = return_fig,
    imsize = as.integer(imsize)
  )

  if(!is.null(nrows)) {
    args$nrows <- as.integer(args$nrows)
  }
  if(!is.null(ncols)) {
    args$ncols <- as.integer(args$ncols)
  }

  do.call(vision$all$get_grid, args)

}

#' @title Clip_remove_empty
#'
#' @description Clip bounding boxes with image border and label background the empty ones
#'
#'
#' @param bbox bbox
#' @param label label
#' @return None
#' @export
clip_remove_empty <- function(bbox, label) {

  vision$all$clip_remove_empty(
    bbox = bbox,
    label = label
  )

}


#' @title Bb_pad
#'
#' @description Function that collect `samples` of labelled bboxes and adds padding with `pad_idx`.
#'
#'
#' @param samples samples
#' @param pad_idx pad_idx
#' @return None
#' @export
bb_pad <- function(samples, pad_idx = 0) {

  vision$all$bb_pad(
    samples = samples,
    pad_idx = as.integer(pad_idx)
  )

}


#' @title PointBlock
#'
#' @description A `TransformBlock` for points in an image
#'
#'
#' @return None
#' @export
PointBlock <- function() {
  invisible(vision$all$PointBlock)
}


#' @title BBoxBlock
#'
#' @description A `TransformBlock` for bounding boxes in an image
#'
#'
#' @return None
#' @export
BBoxBlock <- function() {
  invisible(vision$all$BBoxBlock)
}



#' @title BBoxLblBlock
#'
#' @description A `TransformBlock` for labeled bounding boxes, potentially with `vocab`
#'
#' @return None'
#' @param vocab vocab
#' @param add_na add_na
#'
#' @export
BBoxLblBlock <- function(vocab = NULL, add_na = TRUE) {

  vision$all$BBoxLblBlock(
    vocab = vocab,
    add_na = add_na
  )

}


#' @title SegmentationDataLoaders_from_label_func
#'
#' @description Create from list of `fnames` in `path`s with `label_func`.
#'
#' @param path path
#' @param fnames fnames
#' @param label_func label_func
#' @param valid_pct valid_pct
#' @param seed seed
#' @param codes codes
#' @param item_tfms item_tfms
#' @param batch_tfms batch_tfms
#' @param bs bs
#' @param val_bs val_bs
#' @param shuffle_train shuffle_train
#' @param device device
#' @return None
#' @export
SegmentationDataLoaders_from_label_func <- function(path, fnames, label_func, valid_pct = 0.2,
                            seed = NULL, codes = NULL, item_tfms = NULL,
                            batch_tfms = NULL, bs = 64, val_bs = NULL,
                            shuffle_train = TRUE, device = NULL) {

  args = list(
    path = path,
    fnames = fnames,
    label_func = label_func,
    valid_pct = valid_pct,
    seed = seed,
    codes = codes,
    item_tfms = item_tfms,
    batch_tfms = batch_tfms,
    bs = as.integer(bs),
    val_bs = val_bs,
    shuffle_train = shuffle_train,
    device = device
  )

  if(!is.null(seed)) {
    args$seed = as.integer(args$seed)
  }


  do.call(vision$all$SegmentationDataLoaders$from_label_func, args)

}



