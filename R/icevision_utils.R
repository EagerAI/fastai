

#' @title Show_samples
#'
#'
#' @param samples samples
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
show_samples <- function(samples, class_map = NULL, denormalize_fn = denormalize_imagenet(),
                         display_label = TRUE, display_bbox = TRUE, display_mask = TRUE,
                         ncols = 1, figsize = NULL, show = FALSE, dpi = 100) {

  args = list(
    samples = samples,
    class_map = class_map,
    denormalize_fn = denormalize_fn,
    display_label = display_label,
    display_bbox = display_bbox,
    display_mask = display_mask,
    ncols = as.integer(ncols),
    figsize = figsize,
    show = show
  )

  if(is.list(args$samples)) {
    temp = args$samples
    list_temp = list()
    for (i in 1:length(temp)) {
      list_temp <- append(list_temp, reticulate::r_to_py(list(temp[[i]])) )
    }
    args$samples <- list_temp
  }

  if(is.null(args$class_map))
    args$class_map <- NULL

  if(is.null(args$figsize))
    args$figsize <- NULL

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
#'
#'
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
#'
#'
#' @param samples samples
#' @param preds preds
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
show_preds <- function(samples, preds, class_map = NULL,
                       denormalize_fn = denormalize_imagenet(), display_label = TRUE,
                       display_bbox = TRUE, display_mask = TRUE, ncols = 1,
                       figsize = NULL, show = FALSE, dpi = 100) {

  args <- list(
    samples = samples,
    preds = preds,
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

  do.call(icevision()$show_preds, args)

  tmp_d = gsub(tempdir(), replacement = '/', pattern = '\\', fixed = TRUE)
  fastai2$tabular$all$plt$savefig(paste(tmp_d, 'test.png', sep = '/'), dpi = as.integer(dpi))

  img <- png::readPNG(paste(tmp_d, 'test.png', sep = '/'))
  if(interactive()) {
    try(dev.off(),TRUE)
  }
  grid::grid.raster(img)

}


