


#' @title Crop Time
#'
#' @description Random crops full spectrogram to be length specified in ms by crop_duration
#'
#'
#' @param duration duration
#' @param pad_mode pad_mode
#' @return None
#' @export
CropTime <- function(duration, pad_mode = AudioPadType$Zeros) {

  fastaudio$augment$spectrogram$CropTime(
    duration = duration,
    pad_mode = pad_mode
  )

}


#' @title Mask Freq
#'
#' @description Google SpecAugment frequency masking from https://arxiv.org/abs/1904.08779.
#'
#'
#' @param num_masks number of masks
#' @param size size
#' @param start start point
#' @param val val
#' @return None
#' @export
MaskFreq <- function(num_masks = 1, size = 20, start = NULL, val = NULL) {

  args = list(
    num_masks = as.integer(num_masks),
    size = as.integer(size),
    start = start,
    val = val
  )

  if(!is.null(args[['start']]))
    args[['start']] = as.integer(args[['start']])

  do.call(fastaudio$augment$spectrogram$MaskFreq, args)

}


#' @title MaskTime
#'
#' @description Google SpecAugment time masking from https://arxiv.org/abs/1904.08779.
#'
#'
#' @param num_masks number masks
#' @param size size
#' @param start start point
#' @param val val
#' @return None
#' @export
MaskTime <- function(num_masks = 1, size = 20, start = NULL, val = NULL) {

  args <- list(
    num_masks = as.integer(num_masks),
    size = as.integer(size),
    start = start,
    val = val
  )

  if(!is.null(args[['start']]))
    args[['start']] = as.integer(args[['start']])

  do.call(fastaudio$augment$spectrogram$MaskTime, args)

}


#' @title SGRoll
#'
#' @description Shifts spectrogram along x-axis wrapping around to other side
#'
#'
#' @param max_shift_pct maximum shift percentage
#' @param direction direction
#' @return None
#' @export
SGRoll <- function(max_shift_pct = 0.5, direction = 0) {

  fastaudio$augment$spectrogram$SGRoll(
    max_shift_pct = max_shift_pct,
    direction = as.integer(direction)
  )

}

#' @title Delta
#'
#' @description Creates delta with order 1 and 2 from spectrogram
#'
#' @details and concatenate with the original
#'
#' @param width width
#'
#' @export
Delta <- function(width = 9) {

  fastaudio$augment$spectrogram$Delta(
    width = as.integer(width)
  )

}


#' @title TfmResize
#'
#' @description Temporary fix to allow image resizing transform
#'
#'
#' @param size size
#' @param interp_mode interp_mode
#' @return None
#' @export
TfmResize <- function(size, interp_mode = "bilinear") {

  fastaudio$augment$spectrogram$TfmResize(
    size = size,
    interp_mode = interp_mode
  )

}








