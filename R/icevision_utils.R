

#' @title Show_samples
#'
#' @param dls dataloader
#' @param idx image indices
#' @param class_map class_map
#' @param denormalize_fn denormalize_fn
#' @param display_label display_label
#' @param display_bbox display_bbox
#' @param display_mask display_mask
#' @param ncols ncols
#' @param figsize figsize
#' @param show show
#' @param dpi dots per inch
#' @return None
#' @export
show_samples <- function(dls, idx, class_map = NULL, denormalize_fn = denormalize_imagenet(),
                         display_label = TRUE, display_bbox = TRUE, display_mask = TRUE,
                         ncols = 1, figsize = NULL, show = FALSE, dpi = 100) {

  args = list(
    # we will add samples argument later, below
    #samples = samples,
    class_map = class_map,
    denormalize_fn = denormalize_fn,
    display_label = display_label,
    display_bbox = display_bbox,
    display_mask = display_mask,
    ncols = as.integer(ncols),
    figsize = figsize,
    show = show
  )

  if(is.null(args$class_map))
    args$class_map <- NULL

  if(is.null(args$figsize))
    args$figsize <- NULL


  if(missing(idx))
    args$samples <- reticulate::r_to_py(lapply(1:4, function(x) dls[[x]]))
  else
    args$samples <- reticulate::r_to_py(lapply(idx, function(x) dls[[x]]))

  do.call(icevision()$show_samples, args)

  tmp_d = gsub(tempdir(), replacement = '/', pattern = '\\', fixed = TRUE)
  fastai2$tabular$all$plt$savefig(paste(tmp_d, 'test.png', sep = '/'), dpi = as.integer(dpi))

  img <- png::readPNG(paste(tmp_d, 'test.png', sep = '/'))
  if(interactive()) {
    try(dev.off(),TRUE)
  }
  grid::grid.raster(img)

}


#' @title Denormalize_imagenet
#' @param img img
#' @return None
#' @export
denormalize_imagenet <- function(img) {

  if(missing(img)) {
    icevision()$denormalize_imagenet
  } else {
    icevision()$denormalize_imagenet(
      img = img
    )
  }

}


#' @title Show_preds
#' @param predictions provide list of raw predictions
#' @param idx image indices
#' @param class_map class_map
#' @param denormalize_fn denormalize_fn
#' @param display_label display_label
#' @param display_bbox display_bbox
#' @param display_mask display_mask
#' @param ncols ncols
#' @param figsize figsize
#' @param show show
#' @return None
#' @param dpi dots per inch
#' @export
show_preds <- function(predictions, idx, class_map = NULL,
                       denormalize_fn = denormalize_imagenet(), display_label = TRUE,
                       display_bbox = TRUE, display_mask = TRUE, ncols = 1,
                       figsize = NULL, show = FALSE, dpi = 100) {

  args <- list(
    #add them later to this list
    #samples = samples,
    #preds = preds,
    class_map = class_map,
    denormalize_fn = denormalize_fn,
    display_label = display_label,
    display_bbox = display_bbox,
    display_mask = display_mask,
    ncols = as.integer(ncols),
    figsize = figsize,
    show = show
  )


  if(is.null(args$class_map))
    args$class_map <- NULL

  if(is.null(args$figsize))
    args$figsize <- NULL

  # data extraction

  if(missing(idx)) {
    predicted = reticulate::r_to_py(lapply(1:10, function(x) predictions[[2]][[x]]))
    actual = lapply(1:10, function(x) predictions[[1]][[x]][["img"]])
  } else {
    predicted = reticulate::r_to_py(lapply(idx, function(x) predictions[[2]][[x]]))
    actual = lapply(idx, function(x) predictions[[1]][[x]][["img"]])
  }

  np = reticulate::import('numpy',convert = FALSE)

  for(i in 0:length(predicted)) {
    try(predicted[[i]][["labels"]] <- np$int32(predicted[[i]][["labels"]]), TRUE)
  }

  predicted <- reticulate::r_to_py(predicted)

  args$samples <- actual
  args$preds <- predicted

  ## end

  do.call(icevision()$show_preds, args)

  tmp_d = gsub(tempdir(), replacement = '/', pattern = '\\', fixed = TRUE)
  fastai2$tabular$all$plt$savefig(paste(tmp_d, 'test.png', sep = '/'), dpi = as.integer(dpi))

  img <- png::readPNG(paste(tmp_d, 'test.png', sep = '/'))
  if(interactive()) {
    try(dev.off(),TRUE)
  }
  grid::grid.raster(img)

}


