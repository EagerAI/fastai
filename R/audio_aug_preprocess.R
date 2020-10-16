

#' RemoveType
#' @return module
#' @param ... parameters to pass
#' @export
RemoveType <- NULL


#' @title Remove Silence
#'
#' @description Split signal at points of silence greater than 2*pad_ms
#'
#'
#' @param remove_type remove type from RemoveType module
#' @param threshold threshold point
#' @param pad_ms pad ms
#' @return None
#' @export
RemoveSilence <- function(remove_type = RemoveType$Trim$value, threshold = 20, pad_ms = 20) {

  fastaudio$augment$preprocess$RemoveSilence(
    remove_type = remove_type,
    threshold = as.integer(threshold),
    pad_ms = as.integer(pad_ms)
  )

}


#' @title Resample
#'
#' @description Resample using faster polyphase technique and avoiding FFT computation
#'
#'
#' @param sr_new sr_new
#' @return None
#' @export
Resample <- function(sr_new) {

  fastaudio$augment$preprocess$Resample(
    sr_new = sr_new
  )

}







