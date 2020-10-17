


#' @title Audio_extensions
#' @description get all allowed audio extensions
#' @return vector
#' @export
audio_extensions = function() {
  unlist(fastaudio$core$signal$audio_extensions)
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
    fastaudio$core$signal$get_audio_files
  } else {
    fastaudio$core$signal$get_audio_files(
      path = path,
      recurse = recurse,
      folders = folders
    )
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

  fastaudio$core$signal$AudioGetter(
    suf = suf,
    recurse = recurse,
    folders = folders
  )

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

  fastaudio$core$signal$tar_extract_at_filename(
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

  fastaudio$core$signal$AudioTensor(
    x = x,
    sr = sr
  )

}


#' @title AudioTensor create
#'
#' @description Creates audio tensor from file
#'
#' @param fn function
#' @param cache_folder cache folder
#' @param out out format
#' @param normalization apply normalization or not
#' @param channels_first channels first/last
#' @param num_frames number of frames
#' @param offset offset
#' @param signalinfo signal info
#' @param encodinginfo encoding info
#' @param filetype the type of file
#' @return None
#' @export
AudioTensor_create <- function(fn, cache_folder = NULL, out = NULL,
                               normalization = TRUE, channels_first = TRUE,
                               num_frames = 0, offset = 0, signalinfo = NULL,
                               encodinginfo = NULL, filetype = NULL) {


  if(missing(fn)) {
    fastaudio$core$signal$AudioTensor$create
  } else {
    args = list(
      fn = fn,
      cache_folder = cache_folder,
      out = out,
      normalization = normalization,
      channels_first = channels_first,
      num_frames = as.integer(num_frames),
      offset = as.integer(offset),
      signalinfo = signalinfo,
      encodinginfo = encodinginfo,
      filetype = filetype
    )

    do.call(fastaudio$core$signal$AudioTensor$create, args)
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

  fastaudio$core$signal$OpenAudio(
    items = items
  )

}






