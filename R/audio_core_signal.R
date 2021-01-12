


#' @title Audio_extensions
#' @description get all allowed audio extensions
#' @return vector
#' @export
audio_extensions = function() {
  unlist(fastaudio()$core$signal$audio_extensions)
}


#' @title Get_audio_files
#'
#' @description Get audio files in `path` recursively, only in `folders`, if specified.
#'
#'
#' @param path path
#' @param recurse recursive or not
#' @param folders vector, folders
#' @return None
#' @export
get_audio_files <- function(path, recurse = TRUE, folders = NULL) {

  if(missing(path)) {
    fastaudio()$core$signal$get_audio_files
  } else {
    args = list(
      path = path,
      recurse = recurse,
      folders = folders
    )

    if(is.null(args$folders))
      args$folders <- NULL

    do.call(fastaudio()$core$signal$get_audio_files, args)
  }

}

#' @title AudioGetter
#'
#' @description Create `get_audio_files` partial function that searches path suffix `suf`
#'
#' @details and passes along `kwargs`, only in `folders`, if specified.
#'
#' @param suf suffix
#' @param recurse recursive or not
#' @param folders vector, folders
#' @return None
#' @export
AudioGetter <- function(suf = "", recurse = TRUE, folders = NULL) {

  args = list(
    suf = suf,
    recurse = recurse,
    folders = folders
  )

  if(is.null(args$folders))
    args$folders <- NULL

  do.call(fastaudio()$core$signal$AudioGetter, args)

}


#' @title Tar_extract_at_filename
#'
#' @description Extract `fname` to `dest`/`fname.name` folder using `tarfile`
#'
#'
#' @param fname folder name
#' @param dest destination
#' @return None
#' @export
tar_extract_at_filename <- function(fname, dest) {

  fastaudio()$core$signal$tar_extract_at_filename(
    fname = fname,
    dest = dest
  )

}

#' @title Audio Tensor
#'
#' @description Semantic torch tensor that represents an audio.
#'
#' @param x tensor
#' @param sr sr
#' @return tensor
#' @export
AudioTensor <- function(x, sr = NULL) {

  args = list(
    x = x,
    sr = sr
  )

  if(is.null(args$sr))
    args$sr <- NULL

  do.call(fastaudio()$core$signal$AudioTensor, args)

}


#' @title AudioTensor create
#'
#' @description Creates audio tensor from file
#'
#' @param fn function
#' @param cache_folder cache folder
#' @param normalize apply normalization or not
#' @param channels_first channels first/last
#' @param num_frames number of frames
#' @param frame_offset offset
#' @return None
#' @export
AudioTensor_create <- function(fn, cache_folder = NULL, frame_offset = 0, num_frames = -1,
                               normalize = TRUE, channels_first = TRUE) {


  if(missing(fn)) {
    fastaudio()$core$signal$AudioTensor$create
  } else {
    args = list(
      fn = fn,
      cache_folder = cache_folder,
      frame_offset = as.integer(frame_offset),
      num_frames = as.integer(num_frames),
      normalize = normalize,
      channels_first = channels_first
    )

    if(is.null(args$cache_folder))
      args$cache_folder <- NULL

    do.call(fastaudio()$core$signal$AudioTensor$create, args)
  }

}


#' @title OpenAudio
#'
#' @description Transform that creates AudioTensors from a list of files.
#'
#'
#' @param items vector, items
#' @return None
#'
#' @export
OpenAudio <- function(items) {

  fastaudio()$core$signal$OpenAudio(
    items = items
  )

}






