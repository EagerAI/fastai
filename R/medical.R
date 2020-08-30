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


#' @title Dicom
#'
#' @param img dicom file
#'
#' @export
print.pydicom.dataset.FileDataset <- function(img) {
  cat(img$`__str__`())
}


#' @title Get image matrix
#'
#' @param img dicom file
#' @param type img transformation
#'
#' @export
get_dcm_matrix <- function(img, type = 'raw') {

  obj = medical$dicom_windows

  if (type=='raw') {
    img$pixel_array
  } else if (type=='abdomen_soft') {
    res = obj$abdomen_soft
    img$scaled_px$windowed(res[[1]], res[[2]])$cpu()$numpy()
  } else if (type=='brain') {
    res = obj$brain
    img$scaled_px$windowed(res[[1]], res[[2]])$cpu()$numpy()
  } else if (type=='brain_bone') {
    res = obj$brain_bone
    img$scaled_px$windowed(res[[1]], res[[2]])$cpu()$numpy()
  } else if (type=='brain_soft') {
    res = obj$brain_soft
    img$scaled_px$windowed(res[[1]], res[[2]])$cpu()$numpy()
  } else if (type=='liver') {
    res = obj$liver
    img$scaled_px$windowed(res[[1]], res[[2]])$cpu()$numpy()
  } else if (type=='lungs') {
    res = obj$lungs
    img$scaled_px$windowed(res[[1]], res[[2]])$cpu()$numpy()
  } else if (type=='mediastinum') {
    res = obj$mediastinum
    img$scaled_px$windowed(res[[1]], res[[2]])$cpu()$numpy()
  } else if (type=='spine_bone') {
    res = obj$spine_bone
    img$scaled_px$windowed(res[[1]], res[[2]])$cpu()$numpy()
  } else if (type=='spine_soft') {
    res = obj$spine_soft
    img$scaled_px$windowed(res[[1]], res[[2]])$cpu()$numpy()
  } else if (type=='stroke') {
    res = obj$stroke
    img$scaled_px$windowed(res[[1]], res[[2]])$cpu()$numpy()
  } else if (type=='subdural') {
    res = obj$subdural
    img$scaled_px$windowed(res[[1]], res[[2]])$cpu()$numpy()
  } else if (type=='normalized') {
    img$scaled_px$hist_scaled()$cpu()$numpy()
  }

}









