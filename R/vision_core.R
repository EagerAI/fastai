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

#' @title ToTensor
#'
#' @description Convert item to appropriate tensor class
#'
#' @details
#'
#' @param enc enc
#' @param dec dec
#' @param split_idx split_idx
#' @param order order
#'
#' @export
ToTensor <- function(enc = NULL, dec = NULL, split_idx = NULL, order = NULL) {

  vision$all$ToTensor(
    enc = enc,
    dec = dec,
    split_idx = split_idx,
    order = order
  )

}

#' @title Pipeline
#'
#' @description A pipeline of composed (for encode/decode) transforms, setup with types
#'
#'
#' @param funcs funcs
#' @param split_idx split_idx
#'
#' @export
Pipeline <- function(funcs = NULL, split_idx = NULL) {

  vision$all$Pipeline(
    funcs = funcs,
    split_idx = split_idx
  )

}

#' @title TensorImageBW
#'
#' @param x x
#'
#' @export
TensorImageBW <- function(x) {

  if(missing(x)) {
    invisible(vision$all$TensorImageBW)
  } else {
    vision$all$TensorImageBW(
      x = x
    )
  }

}


#' @title Datasets
#'
#' @description A dataset that creates a list from each `tfms`, passed thru `item_tfms`
#'
#'
#' @param items items
#' @param tfms tfms
#' @param tls tls
#' @param n_inp n_inp
#' @param dl_type dl_type
#' @param use_list use_list
#' @param do_setup do_setup
#' @param split_idx split_idx
#' @param train_setup train_setup
#' @param splits splits
#' @param types types
#' @param verbose verbose
#'
#' @export
Datasets <- function(items = NULL, tfms = NULL, tls = NULL, n_inp = NULL,
                     dl_type = NULL, use_list = NULL, do_setup = TRUE,
                     split_idx = NULL, train_setup = TRUE, splits = NULL,
                     types = NULL, verbose = FALSE) {

  vision$all$Datasets(
    items = items,
    tfms = tfms,
    tls = tls,
    n_inp = n_inp,
    dl_type = dl_type,
    use_list = use_list,
    do_setup = do_setup,
    split_idx = split_idx,
    train_setup = train_setup,
    splits = splits,
    types = types,
    verbose = verbose
  )

}


#' @title Image
#'
#'
#'
#' @param ... parameters to pass
#'
#' @export
Image = function(...) {
  args = list(...)

  do.call(vision$all$PILImage, args)
}


#' @title TfmdDL
#'
#' @description Transformed `DataLoader`
#'
#' @details
#'
#' @param dataset dataset
#' @param bs bs
#' @param shuffle shuffle
#' @param num_workers num_workers
#' @param verbose verbose
#' @param do_setup do_setup
#' @param pin_memory pin_memory
#' @param timeout timeout
#' @param batch_size batch_size
#' @param drop_last drop_last
#' @param indexed indexed
#' @param n n
#' @param device device
#'
#' @export
TfmdDL <- function(dataset, bs = 64, shuffle = FALSE, num_workers = NULL,
                   verbose = FALSE, do_setup = TRUE, pin_memory = FALSE,
                   timeout = 0, batch_size = NULL, drop_last = FALSE,
                   indexed = NULL, n = NULL, device = NULL) {

  vision$all$TfmdDL(
    dataset = dataset,
    bs = as.integer(bs),
    shuffle = shuffle,
    num_workers = num_workers,
    verbose = verbose,
    do_setup = do_setup,
    pin_memory = pin_memory,
    timeout = as.integer(timeout),
    batch_size = batch_size,
    drop_last = drop_last,
    indexed = indexed,
    n = n,
    device = device
  )

}

#' @title PointScaler
#'
#' @description Scale a tensor representing points
#'
#' @param do_scale do_scale
#' @param y_first y_first
#'
#' @export
PointScaler <- function(do_scale = TRUE, y_first = FALSE) {

  vision$all$PointScaler(
    do_scale = do_scale,
    y_first = y_first
  )

}


#' @title tensor
#'
#' @description Like `torch.as_tensor`, but handle lists too, and can pass multiple vector elements directly.
#'
#'
#' @param ... image
#'
#' @export
tensor <- function(...) {

  vision$all$tensor(
    ...
  )

}


#' @title TensorPoint
#'
#' @description Basic type for points in an image
#'
#' @param x x
#'
#' @export
TensorPoint <- function(x) {

  vision$all$TensorPoint(
    x = x
  )

}


#' @title create
#'
#' @description Delegates (`__call__`,`decode`,`setup`) to (`encodes`,`decodes`,`setups`) if `split_idx` matches
#'
#' @param x x
#' @param img_size img_size
#'
#' @export
TensorBBox_create <- function(x, img_size = NULL) {

  if(is.list(img_size)) {
    img_size = as.list(as.integer(unlist(img_size)))
  } else {
    img_size = as.integer(img_size)
  }

  vision$all$TensorBBox$create(
    x = x,
    img_size = img_size
  )

}


#' @title BBoxLabeler
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
BBoxLabeler <- function(enc = NULL, dec = NULL, split_idx = NULL, order = NULL) {

  vision$all$BBoxLabeler(
    enc = enc,
    dec = dec,
    split_idx = split_idx,
    order = order
  )

}


#' @title TensorMultiCategory
#'
#' @param x x
#'
#' @export
TensorMultiCategory <- function(x) {

  vision$all$TensorMultiCategory(
    x = x
  )

}















