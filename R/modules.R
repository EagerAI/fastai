#' Core Pytorch module
#'
#' @export
nn <- NULL


#' Core Pytorch module
#'
#' @export
Module <- NULL


#' Window effect
#'
#' @export
dicom_windows <- NULL


#' Cmap
#'
#' @export
cm <- NULL

#' Colors
#'
#' @export
colors <- NULL

#' Callback
#'
#' @export
Callback <- NULL


#' Kaggle API
#'
#' @export
kg <- NULL


#' Built ins
#'
#' @export
bt <- NULL


#' Functional interface
#'
#' @export
F <- NULL

#' Dicom
#'
#' @export
Dicom <- NULL


#' @title slice
#'
#' @description slice(stop)
#'
#' @details slice(start, stop[, step]) Create a slice object. This is used for extended slicing (e.g. a[0:10:2]).
#'
#'
#' @export
slice <- function(...) {

  args = list(...)

  do.call(bt$slice, args)

}





