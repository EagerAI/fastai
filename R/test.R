

#' @export
"+.torch.nn.modules.container.Sequential" <- function(a, b) {

  res = a$`__dict__`$`_modules`

  if(length(names(res))>0) {
    ll = names(res)
    ll = suppressWarnings(as.numeric(ll))
    ll = ll[!is.na(ll)]
    if(length(ll) > 0) {
      max_ = as.character(max(ll) + 1)
    } else {
      max_ = '0'
    }

  } else {
    max_ = '0'
  }

  if(is.list(b)) {
    max_ = b[[1]]
    b = b[[2]]
  }

  a$add_module(max_, module = b)
  a
}



#' @title Get history
#'
#'
#' @export
to_fastai_training_history <- function(history) {
  structure(class = "fastai_training_history", list(
    history = history
  ))
}


#' @title Plot history
#'
#'
#' @export
plot.to_fastai_training_history <- function(history) {
  plot.ts(history)
}


#' @title show_batch
#'
#' @description
#'
#' @param dls dataloader object
#' @param b b
#' @param max_n max_n
#' @param ctxs ctxs
#' @param show show
#' @param unique unique
#' @export
show_batch <- function(dls, b = NULL, max_n = 9, ctxs = NULL,
                       figsize = c(19.2,10.8),
                       show = TRUE, unique = FALSE, dpi = 90) {

  fastai2$vision$all$plt$close()
  dls$show_batch(
    b = b,
    max_n = as.integer(max_n),
    ctxs = ctxs,
    show = show,
    unique = unique,
    figsize = figsize
  )

  tmp_d = proj_name = gsub(tempdir(), replacement = '/', pattern = '\\', fixed=TRUE)
  fastai2$tabular$all$plt$savefig(paste(tmp_d, 'test.png', sep = '/'), dpi = as.integer(dpi))

  img <- png::readPNG(paste(tmp_d, 'test.png', sep = '/'))
  try(dev.off(),TRUE)
  grid::grid.raster(img)
}



#' @title ClassificationInterpretation_from_learner
#'
#' @description Construct interpretation object from a learner
#'
#' @param learn learn
#' @param ds_idx ds_idx
#' @param dl dl
#' @param act act
#'
#' @export
ClassificationInterpretation_from_learner <- function(learn, ds_idx = 1, dl = NULL, act = NULL) {

  fastai2$vision$all$ClassificationInterpretation$from_learner(
    learn = learn,
    ds_idx = as.integer(ds_idx),
    dl = dl,
    act = act
  )

}


#' @title plot_top_losses
#'
#' @param interp interpretation object
#' @param k k
#' @param largest largest
#' @export
plot_top_losses <- function(interp, k, largest = TRUE, figsize = c(19.2,10.8),
                            ..., dpi = NULL) {

  fastai2$vision$all$plt$close()
  interp$plot_top_losses(
    k = as.integer(k),
    largest = largest,
    figsize = figsize,
    ...
  )

  tmp_d = proj_name = gsub(tempdir(), replacement = '/', pattern = '\\', fixed = TRUE)

  if(is.null(dpi)) {
    fastai2$tabular$all$plt$savefig(paste(tmp_d, 'test.png', sep = '/'))
  } else {
    fastai2$tabular$all$plt$savefig(paste(tmp_d, 'test.png', sep = '/'), dpi = as.integer(dpi))
  }

  img <- png::readPNG(paste(tmp_d, 'test.png', sep = '/'))
  try(dev.off(),TRUE)
  grid::grid.raster(img)
}


#' @title plot_confusion_matrix
#'
#' @description Plot the confusion matrix, with `title` and using `cmap`.
#' @param interp interpretation object
#' @param normalize normalize
#' @param title title
#' @param cmap cmap
#' @param norm_dec norm_dec
#' @param plot_txt plot_txt
#' @importFrom graphics rasterImage
#' @export
plot_confusion_matrix <- function(interp, normalize = FALSE, title = "Confusion matrix",
                                  cmap = "Blues", norm_dec = 2, plot_txt = TRUE,
                                  figsize = c(19.2,10.8),
                                  ..., dpi = 90) {

  fastai2$vision$all$plt$close()
  interp$plot_confusion_matrix(
    normalize = normalize,
    title = title,
    cmap = cmap,
    norm_dec = as.integer(norm_dec),
    plot_txt = plot_txt,
    figsize = figsize,
    dpi = dpi
  )

  tmp_d = proj_name = gsub(tempdir(), replacement = '/', pattern = '\\', fixed = TRUE)
  fastai2$tabular$all$plt$savefig(paste(tmp_d, 'test.png', sep = '/'), dpi = as.integer(dpi))

  img <- png::readPNG(paste(tmp_d, 'test.png', sep = '/'))
  try(dev.off(),TRUE)
  grid::grid.raster(img)

}


#' @title plot_loss
#'
#' @description Plot the losses from `skip_start` and onward
#'
#'
#' @param skip_start skip_start
#' @param with_valid with_valid
#'
#' @export
plot_loss <- function(object, skip_start = 5, with_valid = TRUE, dpi = 100) {

  fastai2$vision$all$plt$close()
  object$recorder$plot_loss(
    skip_start = as.integer(skip_start),
    with_valid = with_valid
  )

  tmp_d = proj_name = gsub(tempdir(), replacement = '/', pattern = '\\', fixed = TRUE)
  fastai2$tabular$all$plt$savefig(paste(tmp_d, 'test.png', sep = '/'), dpi = as.integer(dpi))

  img <- png::readPNG(paste(tmp_d, 'test.png', sep = '/'))
  try(dev.off(),TRUE)
  grid::grid.raster(img)

}

#' @title plot_lr_find
#'
#' @description Plot the result of an LR Finder test (won't work if you didn't do `learn.lr_find()` before)
#'
#'
#' @param skip_end skip_end
#'
#' @export
plot_lr_find <- function(object, skip_end = 5, dpi = 100) {

  fastai2$vision$all$plt$close()
  object$recorder$plot_lr_find(
    skip_end = as.integer(skip_end)
  )

  tmp_d = proj_name = gsub(tempdir(), replacement = '/', pattern = '\\', fixed = TRUE)
  fastai2$tabular$all$plt$savefig(paste(tmp_d, 'test.png', sep = '/'), dpi = as.integer(dpi))

  img <- png::readPNG(paste(tmp_d, 'test.png', sep = '/'))
  try(dev.off(),TRUE)
  grid::grid.raster(img)

}



#' @title most_confused
#'
#' @description Sorted descending list of largest non-diagonal entries of confusion matrix, presented as actual, predicted, number of occurrences.
#'
#'
#' @param min_val min_val
#'
#' @export
most_confused <- function(interp, min_val = 1) {

  res = interp$most_confused(
    min_val = as.integer(min_val)
  )

  res = lapply(1:length(res), function(x) rbind(res[[x]]))

  res = as.data.frame(do.call(rbind,res))

  res

}


#' @title subplots
#'
#'
#' @param nrows nrows
#' @param ncols ncols
#' @param figsize figsize
#' @param imsize imsize
#' @param add_vert add_vert
#'
#' @export
subplots <- function(nrows = 2, ncols = 2, figsize = NULL, imsize = 4, add_vert = 0) {
  fastai2$vision$all$plt$close()
  args <- list(
    nrows = as.integer(nrows),
    ncols = as.integer(ncols),
    figsize = figsize,
    imsize = as.integer(imsize),
    add_vert = as.integer(add_vert)
  )

  do.call(fastai2$medical$imaging$subplots, args)

}

#' @title show
#'
#' @description Adds functionality to view dicom images where each file may have more than 1 frame
#'
#' @details
#'
#' @param frames frames
#' @param scale scale
#'
#' @export
show <- function(img, frames = 1, scale = TRUE, ...) {
  args <- list(
    frames = as.integer(frames),
    scale = scale,
    ...
  )

  if (is.list(scale)) {
    args$scale = reticulate::tuple(scale)
  }

  do.call(img$show, args)

}

#' @title Plot dicom
#'
#'
#'
#' @export
plot.pydicom.dataset.FileDataset = function(x, y, ..., dpi = 100) {
  tmp_d = proj_name = gsub(tempdir(), replacement = '/', pattern = '\\', fixed = TRUE)
  fastai2$tabular$all$plt$savefig(paste(tmp_d, 'test.png', sep = '/'), dpi = as.integer(dpi))

  img <- png::readPNG(paste(tmp_d, 'test.png', sep = '/'))
  try(dev.off(),TRUE)
  grid::grid.raster(img)
  fastai2$vision$all$plt$close()
}



#' @title show_images
#'
#' @description Show all images `ims` as subplots with `rows` using `titles`
#'
#'
#' @param ims ims
#' @param nrows nrows
#' @param ncols ncols
#' @param titles titles
#' @param figsize figsize
#' @param imsize imsize
#' @param add_vert add_vert
#'
#' @export
show_images <- function(ims, nrows = 1, ncols = NULL,
                        titles = NULL, figsize = NULL, imsize = 3, add_vert = 0) {

  args <- list(
    ims = ims,
    nrows = as.integer(nrows),
    ncols = ncols,
    titles = titles,
    figsize = figsize,
    imsize = as.integer(imsize),
    add_vert = as.integer(add_vert)
  )

  if(!is.null(ncols)) {
    args$ncols = as.integer(args$ncols)
  }


  do.call(medical$show_images, args)

}

#' @title Plot tensor
#'
#'
#'
#' @export
plot.list = function(x, y, ..., dpi = 100) {
  tmp_d = proj_name = gsub(tempdir(), replacement = '/', pattern = '\\', fixed = TRUE)
  fastai2$tabular$all$plt$savefig(paste(tmp_d, 'test.png', sep = '/'), dpi = as.integer(dpi))

  img <- png::readPNG(paste(tmp_d, 'test.png', sep = '/'))
  try(dev.off(),TRUE)
  grid::grid.raster(img)
  fastai2$vision$all$plt$close()
}

#' @title uniform_blur2d
#'
#' @description Uniformly apply blurring
#'
#'
#' @param x x
#' @param s s
#'
#' @export
uniform_blur2d <- function(x, s) {

  medical$uniform_blur2d(
    x = x,
    s = as.integer(s)
  )

}



#' @title gauss_blur2d
#'
#' @description Apply gaussian_blur2d kornia filter
#'
#' @details
#'
#' @param x x
#' @param s s
#'
#' @export
gauss_blur2d <- function(x, s) {

   medical$gauss_blur2d(
    x = x,
    s = as.integer(s)
  )

}



#' @title show_results
#'
#' @description Show some predictions on `ds_idx`-th dataset or `dl`
#'
#' @details
#'
#' @param ds_idx ds_idx
#' @param dl dl
#' @param max_n max_n
#' @param shuffle shuffle
#'
#' @export
show_results <- function(object, ds_idx = 1, dl = NULL, max_n = 9, shuffle = TRUE, dpi = 90, ...) {
  fastai2$vision$all$plt$close()

  args <- list(
    ds_idx = as.integer(ds_idx),
    dl = dl,
    max_n = as.integer(max_n),
    shuffle = shuffle,
    ...
  )

  do.call(object$show_results, args)

  tmp_d = proj_name = gsub(tempdir(), replacement = '/', pattern = '\\', fixed = TRUE)
  fastai2$tabular$all$plt$savefig(paste(tmp_d, 'test.png', sep = '/'), dpi = as.integer(dpi))

  img <- png::readPNG(paste(tmp_d, 'test.png', sep = '/'))
  try(dev.off(),TRUE)
  grid::grid.raster(img)
  fastai2$vision$all$plt$close()
}






