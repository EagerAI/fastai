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
get_dcm_matrix <- function(img, type = 'raw', scan = '', size = 50, convert = TRUE) {

  obj = medical$dicom_windows

  fun = function() {
    if(scan=='uniform_blur2d') {
      img = medical$uniform_blur2d(
        img$scaled_px$windowed(res[[1]], res[[2]]), as.integer(size))
    } else if(scan=='gauss_blur2d') {
      img = medical$gauss_blur2d(
        img$scaled_px$windowed(res[[1]], res[[2]]), as.integer(size))
    } else {
      img = img$scaled_px$windowed(res[[1]], res[[2]])
    }

    if(convert) {
      img$cpu()$numpy()
    } else {
      img
    }
  }

  if (type=='raw') {
    img$pixel_array
  } else if (type=='abdomen_soft') {
    res = obj$abdomen_soft
    fun()
  } else if (type=='brain') {
    res = obj$brain
    fun()
  } else if (type=='brain_bone') {
    res = obj$brain_bone
    fun()
  } else if (type=='brain_soft') {
    res = obj$brain_soft
    fun()
  } else if (type=='liver') {
    res = obj$liver
    fun()
  } else if (type=='lungs') {
    res = obj$lungs
    fun()
  } else if (type=='mediastinum') {
    res = obj$mediastinum
    fun()
  } else if (type=='spine_bone') {
    res = obj$spine_bone
    fun()
  } else if (type=='spine_soft') {
    res = obj$spine_soft
    fun()
  } else if (type=='stroke') {
    res = obj$stroke
    fun()
  } else if (type=='subdural') {
    res = obj$subdural
    fun()
  } else if (type=='normalized') {
    if(scan=='uniform_blur2d') {
      img = medical$uniform_blur2d(
        img$scaled_px$hist_scaled(), as.integer(size))
    } else if(scan=='gauss_blur2d') {
      img = medical$gauss_blur2d(
        img$scaled_px$hist_scaled(), as.integer(size))
    } else {
      img = img$hist_scaled()
    }

    if (convert) {
      img$cpu()$numpy()
    } else {
      img
    }
  }

}


#' @title mask_from_blur
#'
#'
#' @param x x
#' @param window window
#' @param sigma sigma
#' @param thresh thresh
#' @param remove_max remove_max
#'
#' @export
mask_from_blur <- function(img, window, sigma = 0.3, thresh = 0.05, remove_max = TRUE) {

  img$mask_from_blur(
    window = list(as.integer(window[[1]]),as.integer(window[[2]])),
    sigma = sigma,
    thresh = thresh,
    remove_max = remove_max
  )

}

#' @title Zoom
#'
#'
#'
#' @param ratio ratio
#'
#' @export
zoom <- function(img, ratio) {

  img$zoom(
    ratio = ratio
  )

}

#' @title Mask2bbox
#'
#'
#' @param mask mask
#'
#' @export
mask2bbox <- function(mask, convert = TRUE) {

  res = medical$mask2bbox(
    mask = mask
  )

  if(convert) {
    lo = res[[0]]$cpu()$numpy()
    hi = res[[1]]$cpu()$numpy()
    list(lo,hi)
  } else {
    res
  }

}

#' @title Abdomen soft
#'
#' @export
win_abdoment_soft <- function() {
  medical$dicom_windows$abdomen_soft
}

#' @title Brain
#'
#' @export
win_brain <- function() {
  medical$dicom_windows$brain
}

#' @title Brain bone
#'
#' @export
win_brain_bone <- function() {
  medical$dicom_windows$brain_bone
}

#' @title Brain soft
#'
#' @export
win_brain_soft <- function() {
  medical$dicom_windows$brain_soft
}

#' @title Liver
#'
#' @export
win_liver <- function() {
  medical$dicom_windows$liver
}

#' @title Lungs
#'
#' @export
win_lungs <- function() {
  medical$dicom_windows$lungs
}
#' @title Mediastinum
#'
#' @export
win_mediastinum <- function() {
  medical$dicom_windows$mediastinum
}

#' @title Spine bone
#'
#' @export
win_spine_bone <- function() {
  medical$dicom_windows$spine_bone
}
#' @title Spine soft
#'
#' @export
win_spine_soft <- function() {
  medical$dicom_windows$spine_soft
}

#' @title Stroke
#'
#' @export
win_stroke <- function() {
  medical$dicom_windows$stroke
}

#' @title Subdural
#'
#' @export
win_subdural <- function() {
  medical$dicom_windows$subdural
}



