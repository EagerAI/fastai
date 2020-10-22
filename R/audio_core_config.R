


#' @title AudioBlock
#'
#' @description A `TransformBlock` for audios
#'
#'
#' @param cache_folder cache folder
#' @param sample_rate sample rate
#' @param force_mono force mono or not
#' @param crop_signal_to int, crop signal
#' @return None
#' @export
AudioBlock <- function(cache_folder = NULL, sample_rate = 16000,
                       force_mono = TRUE, crop_signal_to = NULL) {


  if(missing(cache_folder)) {
    fastaudio$core$config$AudioBlock
  } else {
    fastaudio$core$config$AudioBlock(
      cache_folder = cache_folder,
      sample_rate = as.integer(sample_rate),
      force_mono = force_mono,
      crop_signal_to = crop_signal_to
    )
  }

}



#' @title AudioBlock from folder
#'
#' @description Build a `AudioBlock` from a `path` and caches some intermediary results
#'
#' @param path directory, path
#' @param sample_rate sample rate
#' @param force_mono force mono or not
#' @param crop_signal_to int, crop signal
#' @return None
#' @export
AudioBlock_from_folder <- function(path, sample_rate = 16000,
                                   force_mono = TRUE, crop_signal_to = NULL) {

  fastaudio$core$config$AudioBlock$from_folder(
    path = path,
    sample_rate = as.integer(sample_rate),
    force_mono = force_mono,
    crop_signal_to = crop_signal_to
  )

}


#' @title Preprocess audio folder
#'
#' @description Preprocess audio files in `path` in parallel using `n_workers`
#'
#'
#' @param path directory, path
#' @param folders folders
#' @param output_dir output directory
#' @param sample_rate sample rate
#' @param force_mono force mono or not
#' @param crop_signal_to int, crop signal
#' @return None
#' @export
preprocess_audio_folder <- function(path, folders = NULL, output_dir = NULL,
                                    sample_rate = 16000, force_mono = TRUE,
                                    crop_signal_to = NULL) {

  fastaudio$core$config$preprocess_audio_folder(
    path = path,
    folders = folders,
    output_dir = output_dir,
    sample_rate = as.integer(sample_rate),
    force_mono = force_mono,
    crop_signal_to = crop_signal_to
  )

}


#' @title Preprocess Audio
#'
#' @description Creates an audio tensor and run the basic preprocessing transforms on it.
#'
#' @details Used while preprocessing the audios, this is not a `Transform`.
#'
#' @param sample_rate sample rate
#' @param force_mono force mono or not
#' @param crop_signal_to int, crop signal
#' @return None
#' @export
PreprocessAudio <- function(sample_rate = 16000, force_mono = TRUE, crop_signal_to = NULL) {

  fastaudio$core$config$PreprocessAudio(
    sample_rate = as.integer(sample_rate),
    force_mono = force_mono,
    crop_signal_to = crop_signal_to
  )

}


#' @title BasicMelSpectrogram
#'
#'
#' @param sample_rate sample rate
#' @param n_fft number of fast fourier transforms
#' @param win_length windowing length
#' @param hop_length hopping length
#' @param f_min minimum frequency
#' @param f_max maximum frequency
#' @param pad padding
#' @param n_mels number of mel-spectrograms
#' @param window_fn window function
#' @param power power
#' @param normalized normalized or not
#' @param wkwargs additional arguments
#' @param mel mel-spectrogram or not
#' @param to_db to decibels
#' @return None
#' @export
BasicMelSpectrogram <- function(sample_rate = 16000, n_fft = 400, win_length = NULL,
                                hop_length = NULL, f_min = 0.0, f_max = NULL,
                                pad = 0, n_mels = 128, window_fn = torch$hann_window,
                                power = 2.0, normalized = FALSE, wkwargs = NULL,
                                mel = TRUE, to_db = TRUE) {

  args <- list(
    sample_rate = as.integer(sample_rate),
    n_fft = as.integer(n_fft),
    win_length = win_length,
    hop_length = hop_length,
    f_min = f_min,
    f_max = f_max,
    pad = as.integer(pad),
    n_mels = as.integer(n_mels),
    window_fn = window_fn,
    power = power,
    normalized = normalized,
    wkwargs = wkwargs,
    mel = mel,
    to_db = to_db
  )

  if(!is.null(args[['win_length']]))
    args[['win_length']] = as.integer(args[['win_length']])

  if(!is.null(args[['hop_length']]))
    args[['hop_length']] = as.integer(args[['hop_length']])

  do.call(fastaudio$core$config$AudioConfig$BasicMelSpectrogram, args)

}


#' @title Basic MFCC
#'
#' @param sample_rate sample rate
#' @param n_mfcc number of mel-frequency cepstral coefficients
#' @param dct_type dct type
#' @param norm normalization type
#' @param log_mels apply log to mels
#' @param melkwargs additional arguments for mels
#' @return None
#' @export
BasicMFCC <- function(sample_rate = 16000, n_mfcc = 40, dct_type = 2, norm = "ortho",
                      log_mels = FALSE, melkwargs = NULL) {

  fastaudio$core$config$AudioConfig$BasicMFCC(
    sample_rate = as.integer(sample_rate),
    n_mfcc = as.integer(n_mfcc),
    dct_type = as.integer(dct_type),
    norm = norm,
    log_mels = log_mels,
    melkwargs = melkwargs
  )

}


#' @title BasicSpectrogram
#'
#'
#' @param n_fft number of fast fourier transforms
#' @param win_length windowing length
#' @param hop_length hopping length
#' @param pad padding mode
#' @param window_fn window function
#' @param power power
#' @param normalized normalized or not
#' @param wkwargs additional arguments
#' @param mel mel-spectrogram or not
#' @param to_db to decibels
#' @return None
#' @export
BasicSpectrogram <- function(n_fft = 400, win_length = NULL, hop_length = NULL,
                             pad = 0, window_fn = torch$hann_window, power = 2.0,
                             normalized = FALSE, wkwargs = NULL, mel = FALSE, to_db = TRUE) {

  fastaudio$core$config$AudioConfig$BasicSpectrogram(
    n_fft = as.integer(n_fft),
    win_length = win_length,
    hop_length = hop_length,
    pad = as.integer(pad),
    window_fn = window_fn,
    power = power,
    normalized = normalized,
    wkwargs = wkwargs,
    mel = mel,
    to_db = to_db
  )

}


#' @title Voice
#'
#'
#' @param sample_rate sample rate
#' @param n_fft number of fast fourier transforms
#' @param win_length windowing length
#' @param hop_length hopping length
#' @param f_min minimum frequency
#' @param f_max maximum frequency
#' @param pad padding mode
#' @param n_mels number of mel-spectrograms
#' @param window_fn window function
#' @param power power
#' @param normalized normalized or not
#' @param wkwargs additional arguments
#' @param mel mel-spectrogram or not
#' @param to_db to decibels
#' @return None
#' @export
Voice <- function(sample_rate = 16000, n_fft = 1024, win_length = NULL, hop_length = 128,
                  f_min = 50.0, f_max = 8000.0, pad = 0, n_mels = 128, window_fn = torch$hann_window,
                  power = 2.0, normalized = FALSE, wkwargs = NULL, mel = TRUE, to_db = TRUE) {

  args <- list(
    sample_rate = as.integer(sample_rate),
    n_fft = as.integer(n_fft),
    win_length = win_length,
    hop_length = as.integer(hop_length),
    f_min = f_min,
    f_max = f_max,
    pad = as.integer(pad),
    n_mels = as.integer(n_mels),
    window_fn = window_fn,
    power = power,
    normalized = normalized,
    wkwargs = wkwargs,
    mel = mel,
    to_db = to_db
  )

  if(!is.null(args[['win_length']]))
    args[['win_length']] = as.integer(args[['win_length']])

  do.call(fastaudio$core$config$AudioConfig$Voice, args)

}




