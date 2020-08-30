#' @title get_dicom_files
#'
#' @description Get dicom files in `path` recursively, only in `folders`, if specified.
#'
#'
#' @param path path
#' @param recurse recurse
#' @param folders folders
#'
#' @export
get_dicom_files <- function(path, recurse = TRUE, folders = NULL) {

  medical$get_dicom_files(
    path = path,
    recurse = recurse,
    folders = folders
  )

}

#' @title dcmread
#'
#' @description Open a `DICOM` file
#'
#'
#' @param fn fn
#' @param force force
#'
#' @export
dcmread <- function(fn, force = FALSE) {

  medical$Path$dcmread(
    fn = fn,
    force = force
  )

}





