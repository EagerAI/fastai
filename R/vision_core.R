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
Image_resize <- function(img, size, resample = 3, box = NULL, reducing_gap = NULL) {

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


#' @title to_image
#'
#' @description Convert a tensor or array to a PIL int8 Image
#'
#'
#' @param x x
#'
#' @export
to_image <- function(x) {

  vision$all$to_image(
    x = x
  )

}

#' @title load_image
#'
#' @description Open and load a `PIL.Image` and convert to `mode`
#'
#'
#' @param fn fn
#' @param mode mode
#'
#' @export
load_image <- function(fn, mode = NULL) {

  vision$all$load_image(
    fn = fn,
    mode = mode
  )

}


#' @title image2tensor
#'
#' @description Transform image to byte tensor in `c*h*w` dim order.
#'
#'
#' @param img img
#'
#' @export
image2tensor <- function(img) {

  vision$all$image2tensor(
    img = img
  )

}

#' @title create
#'
#' @description Open an `Image` from path `fn`
#'
#' @param fn fn
#'
#' @export
Image_create <- function(fn) {

  vision$all$PILImage$create(
    fn = fn
  )

}


#' @title create
#'
#' @description Delegates (`__call__`,`decode`,`setup`) to (`encodes`,`decodes`,`setups`) if `split_idx` matches
#'
#' @details
#'
#' @param enc enc
#' @param dec dec
#' @param split_idx split_idx
#' @param order order
#'
#' @export
Mask_create <- function(enc = NULL, dec = NULL, split_idx = NULL, order = NULL) {

  if(is.null(enc) & is.null(dec) & is.null(split_idx) & is.null(order)) {
    invisible(vision$all$PILMask$create)
  } else {
    args = list(
      enc = enc,
      dec = dec,
      split_idx = split_idx,
      order = order
    )

    if(!is.null(split_idx)) {
      args$split_idx = as.integer(args$split_idx)
    }

    do.call(vision$all$PILMask$create, args)
  }

}


#' @title Transform
#'
#' @description Delegates (`__call__`,`decode`,`setup`) to (`encodes`,`decodes`,`setups`) if `split_idx` matches
#'
#'
#' @param enc enc
#' @param dec dec
#' @param split_idx split_idx
#' @param order order
#'
#' @export
Transform <- function(enc = NULL, dec = NULL, split_idx = NULL, order = NULL) {

  args <- list(
    enc = enc,
    dec = dec,
    split_idx = split_idx,
    order = order
  )

  if(!is.null(split_idx)) {
    args$split_idx = as.integer(args$split_idx)
  }

  do.call(vision$all$Transform, args)

}

#' @title create
#'
#' @description Open an `Image` from path `fn`
#'
#' @param fn fn
#'
#' @export
ImageBW_create <- function(fn) {

  vision$all$PILImageBW$create(
    fn = fn
  )

}



#' @title create
#'
#' @description Delegates (`__call__`,`decode`,`setup`) to (`encodes`,`decodes`,`setups`) if `split_idx` matches
#'
#' @details
#'
#' @param enc enc
#' @param dec dec
#' @param split_idx split_idx
#' @param order order
#'
#' @export
TensorPoint_create <- function(enc = NULL, dec = NULL, split_idx = NULL, order = NULL) {

  if(is.null(enc) & is.null(dec) & is.null(split_idx) & is.null(order)) {
    invisible(vision$all$TensorPoint$create)
  } else {
    args = list(
      enc = enc,
      dec = dec,
      split_idx = split_idx,
      order = order
    )

    if(!is.null(split_idx)) {
      args$split_idx = as.integer(args$split_idx)
    }

    do.call(vision$all$TensorPoint$create, args)
  }
}


#' @title TensorBBox
#'
#' @description Basic type for a tensor of bounding boxes in an image
#'
#' @param x x
#'
#' @export
TensorBBox <- function(x) {

  vision$all$TensorBBox(
    x = x
  )

}

#' @title LabeledBBox
#'
#' @description Basic type for a list of bounding boxes in an image
#'
#'
#' @param items items
#'
#' @export
LabeledBBox <- function( ...) {

  vision$all$LabeledBBox(
    ...
  )

}















