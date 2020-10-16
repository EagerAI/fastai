


#' AudioSpectrogram
#' @return module
#' @param ... parameters to pass
#' @export
AudioSpectrogram <- NULL



#' @title AudioToSpec from cfg
#'
#' @description Creates AudioToSpec from configuration file
#'
#' @param audio_cfg audio_cfg
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
#' @param mel mel
#' @param to_db to_db
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
#' @param sample_rate sample_rate
#' @param n_mfcc n_mfcc
#' @param dct_type dct_type
#' @param norm norm
#' @param log_mels log_mels
#' @param melkwargs melkwargs
#' @return None
#' @export
AudioToMFCC <- function(sample_rate = 16000, n_mfcc = 40, dct_type = 2,
                        norm = "ortho", log_mels = FALSE, melkwargs = NULL) {

  fastaudio$core$spectrogram$AudioToMFCC(
    sample_rate = as.integer(sample_rate),
    n_mfcc = as.integer(n_mfcc),
    dct_type = as.integer(dct_type),
    norm = norm,
    log_mels = log_mels,
    melkwargs = melkwargs
  )

}



#' @title AudioToMFCC from cfg
#'
#' @description Creates AudioToMFCC from configuration file
#'
#' @param audio_cfg audio_cfg
#' @return None
#' @export
AudioToMFCC_from_cfg <- function(audio_cfg) {

  fastaudio$core$spectrogram$AudioToMFCC$from_cfg(
    audio_cfg = audio_cfg
  )

}





