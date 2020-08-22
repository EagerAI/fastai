#' @title From_name_re
#'
#' @description Create from the name attrs of `fnames` in `path`s with re expression `pat`
#'
#' @param path path
#' @param fnames fnames
#' @param pat pat
#' @param bs bs
#' @param val_bs val_bs
#' @param shuffle_train shuffle_train
#' @param device device
#' @param item_tfms item_tfms
#' @param batch_tfms batch_tfms
#'
#' @export
ImageDataLoaders_from_name_re <- function(path, fnames, pat, bs = 64,
                                          val_bs = NULL, shuffle_train = TRUE,
                                          device = NULL, item_tfms = NULL,
                                          batch_tfms = NULL,
                                          ...) {

  args <- list(
    path = path,
    fnames = fnames,
    pat = pat,
    bs = as.integer(bs),
    val_bs = val_bs,
    shuffle_train = shuffle_train,
    device = device,
    item_tfms = item_tfms,
    batch_tfms = batch_tfms,
    ...
  )

  if(!is.null(args$batch_tfms)) {
    args$batch_tfms <- unlist(args$batch_tfms)
  }
  do.call(vision$all$ImageDataLoaders$from_name_re,args)

}

#' @title Get_image_files
#'
#' @description Get image files in `path` recursively, only in `folders`, if specified.
#'
#'
#' @param path path
#' @param recurse recurse
#' @param folders folders
#'
#' @export
get_image_files <- function(path, recurse = TRUE, folders = NULL) {

  args <- list(
    path = path,
    recurse = recurse,
    folders = folders
  )

  do.call(vision$all$get_image_files, args)

}

#' @title Fit one cycle
#'
#' @param n_epoch n_epoch
#' @param ... parameters to pass
#'
#' @export
fit_one_cycle = function(object, n_epoch, ...) {
  args = list(n_epoch = as.integer(n_epoch),
              ...)
  do.call(object$fit_one_cycle, args)
}


#' @title Random image batch
#'
#' @description for visualization
#' @param object dataloaders object
#' @param regex for img titles
#' @export
random_batch <- function(object, regex = "[A-z]+_") {
  batch = object$one_batch()
  indices = batch[[2]]$cpu()$numpy()+1

  object$train_ds$items[indices] -> img_p
  lapply(1:length(img_p), function(x) as.character(img_p[[x]])) -> img_p
  names(img_p) = unlist(img_p)
  names(img_p) = trimws(gsub(pattern="_",replacement=' ', regmatches(img_p,regexpr(regex,names(img_p))) ))
  img_p
}









