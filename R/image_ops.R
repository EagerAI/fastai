#' @title open_image
#'
#' @description Return `Image` object created from image in file `fn`.
#'
#' @details
#'
#' @param fn fn
#' @param div div
#' @param convert_mode convert_mode
#' @param cls cls
#' @param after_open after_open
#'
#' @export
open_image <- function(fn, div = TRUE, convert_mode = "RGB", after_open = NULL, ...) {

  vision$image$open_image(
    fn = fn,
    div = div,
    convert_mode = convert_mode,
    after_open = after_open,
    ...
  )

}



#' @title open_mask
#'
#' @description Return `ImageSegment` object create from mask in file `fn`. If `div`, divides pixel values by 255.
#'
#' @details
#'
#' @param fn fn
#' @param div div
#' @param convert_mode convert_mode
#' @param after_open after_open
#'
#' @export
open_mask <- function(fn, div = FALSE, convert_mode = "L", after_open = NULL) {

  vision$image$open_mask(
    fn = fn,
    div = div,
    convert_mode = convert_mode,
    after_open = after_open
  )

}

#' @title open_mask_rle
#'
#' @description Return `ImageSegment` object create from run-length encoded string in `mask_lre` with size in `shape`.
#'
#'
#' @param mask_rle mask_rle
#' @param shape shape
#'
#' @export
open_mask_rle <- function(mask_rle, shape) {

  vision$image$open_mask_rle(
    mask_rle = mask_rle,
    shape = shape
  )

}

#' @title ImagePoints
#'
#' @description Support applying transforms to a `flow` of points.
#'
#' @details
#'
#' @param flow flow
#' @param scale scale
#' @param y_first y_first
#'
#' @export
ImagePoints <- function(flow, scale = TRUE, y_first = TRUE) {

  python_function_result <- vision$image$ImagePoints(
    flow = flow,
    scale = scale,
    y_first = y_first
  )

}







