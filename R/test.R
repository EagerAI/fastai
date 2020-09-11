

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
#' @importFrom graphics rasterImage
#' @export
show_batch <- function(dls, b = NULL, max_n = 9, ctxs = NULL,
                       figsize = c(19.2,10.8),
                       show = TRUE, unique = FALSE, dpi = 90) {

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
  lim <- par()
  plot.new()
  rasterImage(img, lim$usr[1], lim$usr[3], lim$usr[2], lim$usr[4])
  invisible(file.remove(paste(tmp_d, 'test.png', sep = '/')))
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
#' @importFrom graphics rasterImage
#' @export
plot_top_losses <- function(interp, k, largest = TRUE, figsize = c(19.2,10.8),
                            ..., dpi = 90) {

  interp$plot_top_losses(
    k = as.integer(k),
    largest = largest,
    ...
  )

  tmp_d = proj_name = gsub(tempdir(), replacement = '/', pattern = '\\', fixed = TRUE)
  fastai2$tabular$all$plt$savefig(paste(tmp_d, 'test.png', sep = '/'), dpi = as.integer(dpi))

  img <- png::readPNG(paste(tmp_d, 'test.png', sep = '/'))
  lim <- par()
  plot.new()
  rasterImage(img, lim$usr[1], lim$usr[3], lim$usr[2], lim$usr[4])
  invisible(file.remove(paste(tmp_d, 'test.png', sep = '/')))
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

  interp$plot_confusion_matrix(
    normalize = normalize,
    title = title,
    cmap = cmap,
    norm_dec = as.integer(norm_dec),
    plot_txt = plot_txt
  )

  tmp_d = proj_name = gsub(tempdir(), replacement = '/', pattern = '\\', fixed = TRUE)
  fastai2$tabular$all$plt$savefig(paste(tmp_d, 'test.png', sep = '/'), dpi = as.integer(dpi))

  img <- png::readPNG(paste(tmp_d, 'test.png', sep = '/'))
  lim <- par()
  plot.new()
  rasterImage(img, lim$usr[1], lim$usr[3], lim$usr[2], lim$usr[4])
  invisible(file.remove(paste(tmp_d, 'test.png', sep = '/')))

}






