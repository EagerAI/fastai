

#' @title AudioPadType
#'
#' @description An enumeration.
#'
#' @param ... parameters to pass
#' @return module
#' @export
AudioPadType <- NULL


#' @title Resize Signal
#'
#' @description Crops signal to be length specified in ms by duration, padding if needed
#'
#'
#' @param duration int, duration
#' @param pad_mode padding mode
#' @return None
#' @export
ResizeSignal <- function(duration, pad_mode = AudioPadType$Zeros) {

  fastaudio$augment$signal$ResizeSignal(
    duration = duration,
    pad_mode = pad_mode
  )

}

#' @title Signal Shifter
#'
#' @description Randomly shifts the audio signal by `max_pct` %.
#'
#' @details direction must be -1(left) 0(bidirectional) or 1(right).
#'
#' @param p probability
#' @param max_pct max percentage
#' @param max_time maximum time
#' @param direction direction
#' @param roll roll or not
#' @return None
#' @export
SignalShifter <- function(p = 0.5, max_pct = 0.2, max_time = NULL, direction = 0, roll = FALSE) {

  fastaudio$augment$signal$SignalShifter(
    p = p,
    max_pct = max_pct,
    max_time = max_time,
    direction = as.integer(direction),
    roll = roll
  )

}


#' @title Noise Color
#'
#' @param ... parameters to pass
#'
#' @return module
#' @export
NoiseColor <- NULL


#' @title Add Noise
#'
#' @description Adds noise of specified color and level to the audio signal
#'
#'
#' @param noise_level noise level
#' @param color int, color
#' @return None
#' @export
AddNoise <- function(noise_level = 0.05, color = 0) {

  fastaudio$augment$signal$AddNoise(
    noise_level = noise_level,
    color = as.integer(color)
  )

}

#' @title Change Volume
#'
#' @description Changes the volume of the signal
#'
#'
#' @param p probability
#' @param lower lower bound
#' @param upper upper bound
#' @return None
#' @export
ChangeVolume <- function(p = 0.5, lower = 0.5, upper = 1.5) {

  fastaudio$augment$signal$ChangeVolume(
    p = p,
    lower = lower,
    upper = upper
  )

}


#' @title Signal Cutout
#'
#' @description Randomly zeros some portion of the signal
#'
#'
#' @param p probability
#' @param max_cut_pct max cut percentage
#' @return None
#' @export
SignalCutout <- function(p = 0.5, max_cut_pct = 0.15) {

  fastaudio$augment$signal$SignalCutout(
    p = p,
    max_cut_pct = max_cut_pct
  )

}


#' @title Signal Loss
#'
#' @description Randomly loses some portion of the signal
#'
#'
#' @param p probability
#' @param max_loss_pct max loss percentage
#' @return None
#' @export
SignalLoss <- function(p = 0.5, max_loss_pct = 0.15) {

  fastaudio$augment$signal$SignalLoss(
    p = p,
    max_loss_pct = max_loss_pct
  )

}


#' @title Downmix Mono
#'
#' @description Transform multichannel audios into single channel
#'
#' @param enc encoder
#' @param dec decoder
#' @param split_idx split by index
#' @param order order, by default is NULL
#' @return None
#' @export
DownmixMono <- function(enc = NULL, dec = NULL, split_idx = NULL, order = NULL) {

  args = list(
    enc = enc,
    dec = dec,
    split_idx = split_idx,
    order = order
  )

  if(!is.null(args[['split_idx']])) {
    args[['split_idx']] = as.integer(args[['split_idx']])
  }

  do.call(fastaudio$augment$signal$DownmixMono, args)

}















