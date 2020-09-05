#' @title open
#'
#' @description Opens and identifies the given image file.
#'
#' @param fp fp
#' @param mode mode
#'
#' @export
Image_open <- function(fp, mode = "r") {

  vision$all$Image$open(
    fp = fp,
    mode = mode
  )

}


#' @title resize
#'
#' @description Returns a resized copy of this image.
#'
#' @param img image
#'
#' @param size size
#' @param resample resample
#' @param box box
#' @param reducing_gap reducing_gap
#'
#' @export
resize <- function(img, size, resample = 3, box = NULL, reducing_gap = NULL) {

  args <- list(
    size = size,
    resample = as.integer(resample),
    box = box,
    reducing_gap = reducing_gap
  )

  if(is.list(size)) {
    args$size = as.list(as.integer(unlist(args$size)))
  }

  if(is.vector(size)) {
    args$size = as.integer(args$size)
  }

  do.call(img$resize, args)

}


#' @title n_px
#'
#' @description int(x=0) -> integer
#' @param img image
#'
#' @export
n_px = function(img) {
  img$n_px
}


#' @title Shape
#'
#'
#' @param img image
#'
#' @export
shape = function(img) {
  unlist(img$shape)
}


#' @title Aspect
#'
#'
#' @param img image
#'
#' @export
aspect = function(img) {
  img$aspect
}


#' @title reshape
#'
#' @description `resize` `x` to `(w,h)`
#'
#' @details
#'
#' @param x x
#' @param h h
#' @param w w
#' @param resample resample
#'
#' @export
reshape <- function(x, h, w, resample = 0) {

  args <- list(
    h = as.integer(h),
    w = as.integer(w),
    resample = as.integer(resample)
  )

  do.call(x$reshape, args)

}


#' @title to_bytes_format
#'
#' @description Convert to bytes, default to PNG format
#'
#' @details
#'
#' @param im im
#' @param format format
#'
#' @export
to_bytes_format <- function(img, format = "png") {

  img$to_bytes_format(
    format = format
  )

}


#' @title to_thumb
#'
#' @description Same as `thumbnail`, but uses a copy
#'
#'
#' @param h h
#' @param w w
#'
#' @export
to_thumb <- function(h, w = NULL) {

  img$to_thumb(
    h = h,
    w = w
  )

}



#' @title resize_max
#'
#' @description `resize` `x` to `max_px`, or `max_h`, or `max_w`
#'
#' @details
#'
#' @param x x
#' @param resample resample
#' @param max_px max_px
#' @param max_h max_h
#' @param max_w max_w
#'
#' @export
resize_max <- function(img, resample = 0, max_px = NULL, max_h = NULL, max_w = NULL) {

  args <- list(
    resample = as.integer(resample),
    max_px = max_px,
    max_h = max_h,
    max_w = max_w
  )



  do.call(img$resize_max, args)

}











