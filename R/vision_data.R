
#' @title Get_grid
#'
#' @description Return a grid of `n` axes, `rows` by `cols`
#'
#'
#' @param n n
#' @param nrows number of rows
#' @param ncols number of columns
#' @param add_vert add vertical
#' @param figsize figure size
#' @param double double
#' @param title title
#' @param return_fig return figure or not
#' @param imsize image size
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

  do.call(vision()$all$get_grid, args)

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

  vision()$all$clip_remove_empty(
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
#' @param pad_idx pad index
#' @return None
#' @export
bb_pad <- function(samples, pad_idx = 0) {

  vision()$all$bb_pad(
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
  invisible(vision()$all$PointBlock)
}


#' @title BBoxBlock
#'
#' @description A `TransformBlock` for bounding boxes in an image
#'
#'
#' @return None
#' @export
BBoxBlock <- function() {
  invisible(vision()$all$BBoxBlock)
}



#' @title BBoxLblBlock
#'
#' @description A `TransformBlock` for labeled bounding boxes, potentially with `vocab`
#'
#' @return None'
#' @param vocab vocabulary
#' @param add_na add NA
#'
#' @examples
#'
#' \dontrun{
#'
#' URLs_COCO_TINY()
#'
#' c(images, lbl_bbox) %<-% get_annotations('coco_tiny/train.json')
#' timg = Transform(ImageBW_create)
#' idx = 49
#' c(coco_fn,bbox) %<-% list(paste('coco_tiny/train',images[[idx]],sep = '/'),
#'                           lbl_bbox[[idx]])
#' coco_img = timg(coco_fn)
#'
#' tbbox = LabeledBBox(TensorBBox(bbox[[1]]), bbox[[2]])
#'
#' coco_bb = function(x) {
#' TensorBBox_create(bbox[[1]])
#' }
#'
#' coco_lbl = function(x) {
#'   bbox[[2]]
#' }
#'
#' coco_dsrc = Datasets(c(rep(coco_fn,10)),
#'                      list(Image_create(), list(coco_bb),
#'                           list( coco_lbl, MultiCategorize(add_na = TRUE) )
#'                      ), n_inp = 1)
#'
#' coco_tdl = TfmdDL(coco_dsrc, bs = 9,
#'                   after_item = list(BBoxLabeler(), PointScaler(),
#'                                     ToTensor()),
#'                   after_batch = list(IntToFloatTensor(), aug_transforms())
#' )
#'
#' coco_tdl %>% show_batch(dpi = 200)
#'
#' }
#'
#' @export
BBoxLblBlock <- function(vocab = NULL, add_na = TRUE) {

  vision()$all$BBoxLblBlock(
    vocab = vocab,
    add_na = add_na
  )

}


#' @title SegmentationDataLoaders_from_label_func
#'
#' @description Create from list of `fnames` in `path`s with `label_func`.
#'
#' @param path path
#' @param fnames file names
#' @param label_func label function
#' @param valid_pct validation percentage
#' @param seed seed
#' @param codes codes
#' @param item_tfms item transformations
#' @param batch_tfms batch transformations
#' @param bs batch size
#' @param val_bs validation batch size
#' @param shuffle_train shuffle train
#' @param device device name
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

  strings = c('codes', 'item_tfms',
              'batch_tfms', 'device')

  for(i in 1:length(strings)) {
    if(is.null(args[[strings[i]]]))
      args[[strings[i]]] <- NULL
  }

  if(is.null(args$seed)) {
    args$seed <- NULL
  } else {
    args$seed <- as.integer(args$seed)
  }

  if(is.null(args$val_bs)) {
    args$val_bs <- NULL
  } else {
    args$val_bs <- as.integer(args$val_bs)
  }


  do.call(vision()$all$SegmentationDataLoaders$from_label_func, args)

}



