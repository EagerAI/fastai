


#' @title AudioSpectrogram
#' @return module
#' @param ... parameters to pass
#' @export
AudioSpectrogram <- NULL



#' @title AudioToSpec from cfg
#'
#' @description Creates AudioToSpec from configuration file
#'
#' @param audio_cfg audio configuration
#' @return None
#' @export
AudioToSpec_from_cfg <- function(audio_cfg) {

  fastaudio$core$spectrogram$AudioToSpec$from_cfg(
    audio_cfg = audio_cfg
  )

}


#' @title Spectrogram Transformer
#'
#' @description Creates a factory for creating AudioToSpec
#'
#' @details transforms with different parameters
#'
#' @param mel mel-spectrogram or not
#' @param to_db to decibels
#' @return None
#' @export
SpectrogramTransformer <- function(mel = TRUE, to_db = TRUE) {

  fastaudio$core$spectrogram$SpectrogramTransformer(
    mel = mel,
    to_db = to_db
  )

}


#' @title AudioToMFCC
#'
#' @description Transform to create MFCC features from audio tensors.
#'
#'
#' @param sample_rate sample rate
#' @param n_mfcc number of mel-frequency cepstral coefficients
#' @param dct_type dct type
#' @param norm normalization type
#' @param log_mels apply log to mels
#' @param melkwargs additional arguments for mels
#' @return None
#' @export
AudioToMFCC <- function(sample_rate = 16000, n_mfcc = 40, dct_type = 2,
                        norm = "ortho", log_mels = FALSE, melkwargs = NULL) {

  args = list(
    sample_rate = as.integer(sample_rate),
    n_mfcc = as.integer(n_mfcc),
    dct_type = as.integer(dct_type),
    norm = norm,
    log_mels = log_mels,
    melkwargs = melkwargs
  )

  strings = c('sample_rate', 'n_fft', 'hop_length', 'win_length', 'pad', 'n_mels')

  for (i in 1:length(strings)) {
    if(!is.null(args[['melkwargs']][[strings[i]]])) {
      args[['melkwargs']][[strings[i]]] = as.integer(args[['melkwargs']][[strings[i]]])
    }
  }

  do.call(fastaudio$core$spectrogram$AudioToMFCC, args)

}



#' @title AudioToMFCC from cfg
#'
#' @description Creates AudioToMFCC from configuration file
#'
#' @param audio_cfg audio configuration
#' @return None
#' @export
AudioToMFCC_from_cfg <- function(audio_cfg) {

  fastaudio$core$spectrogram$AudioToMFCC$from_cfg(
    audio_cfg = audio_cfg
  )

}





