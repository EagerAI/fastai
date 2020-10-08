#' Core Pytorch module
#' @return module
#' @export
nn <- NULL


#' Core Pytorch module
#' @return module
#' @export
Module <- NULL


#' Window effect
#' @return module
#' @export
dicom_windows <- NULL


#' Cmap
#' @return module
#' @export
cm <- NULL

#' Colors
#' @return module
#' @export
colors <- NULL

#' Callback
#' @return module
#' @param ... additional arguments
#' @export
Callback <- NULL


#' Kaggle API
#' @return module
#' @export
kg <- NULL


#' Built ins
#' @return module
#' @export
bt <- NULL


#' Functional interface
#' @return module
#' @export
F <- NULL

#' Dicom
#' @return module
#' @param ... additional arguments
#' @export
Dicom <- NULL


#' @title Slice
#'
#' @param ... additional arguments
#' @details slice(start, stop[, step]) Create a slice object. This is used for extended slicing (e.g. a[0:10:2]).
#'
#' @return sliced object
#' @export
slice <- function(...) {

  args = list(...)

  do.call(bt$slice, args)

}





