#' @title Image_open
#'
#' @description Opens and identifies the given image file.
#'
#' @param fp fp
#' @param mode mode
#' @return None
#' @export
Image_open <- function(fp, mode = "r") {

  vision()$all$Image$open(
    fp = fp,
    mode = mode
  )

}


#' @title Resize
#'
#' @description Returns a resized copy of this image.
#'
#' @param img image
#'
#' @param size size
#' @param resample resample
#' @param box box
#' @param reducing_gap reducing_gap
#' @return None
#' @export
Image_resize <- function(img, size, resample = 3, box = NULL, reducing_gap = NULL) {

  args <- list(
    size = size,
    resample = as.integer(resample),
    box = box,
    reducing_gap = reducing_gap
  )

  strings = c('box', 'reducing_gap')

  for(i in 1:length(strings)) {
    if(is.null(args[[strings[i]]]))
      args[[strings[i]]] <- NULL
  }

  if(is.list(size)) {
    args$size = as.list(as.integer(unlist(args$size)))
  }

  if(is.vector(size)) {
    args$size = as.integer(args$size)
  }

  do.call(img$resize, args)

}


#' @title N_px
#'
#' @description int(x=0) -> integer
#' @param img image
#' @return None
#' @export
n_px = function(img) {
  img$n_px
}


#' @title Shape
#' @param img image
#' @return None
#' @export
shape = function(img) {
  unlist(img$shape)
}


#' @title Aspect
#' @param img image
#' @return None
#' @export
aspect = function(img) {
  img$aspect
}


#' @title Reshape
#'
#' @description resize x to (w,h)
#'
#'
#' @param x tensor
#' @param h height
#' @param w width
#' @param resample resample value
#' @return None
#' @export
reshape <- function(x, h, w, resample = 0) {

  args <- list(
    h = as.integer(h),
    w = as.integer(w),
    resample = as.integer(resample)
  )

  do.call(x$reshape, args)

}


#' @title To_bytes_format
#'
#' @description Convert to bytes, default to PNG format
#'
#'
#' @param img image
#' @param format format
#' @return None
#' @export
to_bytes_format <- function(img, format = "png") {

  img$to_bytes_format(
    format = format
  )

}


#' @title To_thumb
#'
#' @description Same as `thumbnail`, but uses a copy
#'
#' @param img image
#' @param h height
#' @param w width
#' @return None
#' @export
to_thumb <- function(img, h, w = NULL) {

  img$to_thumb(
    h = h,
    w = w
  )

}



#' @title Resize_max
#'
#' @description `resize` `x` to `max_px`, or `max_h`, or `max_w`
#'
#'
#' @param img image
#' @param resample resample value
#' @param max_px max px
#' @param max_h max height
#' @param max_w max width
#' @return None
#' @export
resize_max <- function(img, resample = 0, max_px = NULL, max_h = NULL, max_w = NULL) {

  args <- list(
    resample = as.integer(resample),
    max_px = max_px,
    max_h = max_h,
    max_w = max_w
  )

  strings = c('max_px', 'max_h', 'max_w')

  for(i in 1:length(strings)) {
    if(is.null(args[[strings[i]]]))
      args[[strings[i]]] <- NULL
  }

  do.call(img$resize_max, args)

}


#' @title To_image
#'
#' @description Convert a tensor or array to a PIL int8 Image
#'
#'
#' @param x tensor
#' @return None
#' @export
to_image <- function(x) {

  vision()$all$to_image(
    x = x
  )

}

#' @title Load_image
#'
#' @description Open and load a `PIL.Image` and convert to `mode`
#'
#'
#' @param fn file name
#' @param mode mode
#' @return None
#' @export
load_image <- function(fn, mode = NULL) {

  vision()$all$load_image(
    fn = fn,
    mode = mode
  )

}


#' @title Image2tensor
#'
#' @description Transform image to byte tensor in `c*h*w` dim order.
#'
#'
#' @param img image
#' @return None
#' @export
image2tensor <- function(img) {

  vision()$all$image2tensor(
    img = img
  )

}

#' @title Image_create
#'
#' @description Open an `Image` from path `fn`
#'
#' @param fn file name
#' @return None
#' @export
Image_create <- function(fn) {

  if(missing(fn)) {
    invisible(vision()$all$PILImage$create)
  } else {
    vision()$all$PILImage$create(
      fn = fn
    )
  }

}


#' @title Mask_create
#'
#' @description Delegates (`__call__`,`decode`,`setup`) to (`encodes`,`decodes`,`setups`) if `split_idx` matches
#'
#'
#' @param enc encoder
#' @param dec decoder
#' @param split_idx split by index
#' @param order order
#' @return None
#' @export
Mask_create <- function(enc = NULL, dec = NULL, split_idx = NULL, order = NULL) {

  if(is.null(enc) & is.null(dec) & is.null(split_idx) & is.null(order)) {
    invisible(vision()$all$PILMask$create)
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

    do.call(vision()$all$PILMask$create, args)
  }

}


#' @title Transform
#'
#' @description Delegates (`__call__`,`decode`,`setup`) to (`encodes`,`decodes`,`setups`) if `split_idx` matches
#'
#'
#' @param enc encoder
#' @param dec decoder
#' @param split_idx split by index
#' @param order order
#' @return None
#' @export
Transform <- function(enc = NULL, dec = NULL, split_idx = NULL, order = NULL) {

  args <- list(
    enc = enc,
    dec = dec,
    split_idx = split_idx,
    order = order
  )

  if(!is.null(split_idx)) {
    args$split_idx = as.integer(unlist(args$split_idx))
  }

  if(is.null(enc) & is.null(dec) & is.null(split_idx) & is.null(order)) {
    vision()$all$Transform
  } else {
    do.call(vision()$all$Transform, args)
  }

}

#' @title ImageBW_create
#'
#' @description Open an `Image` from path `fn`
#'
#' @param fn file name
#' @return None
#' @export
ImageBW_create <- function(fn) {

  vision()$all$PILImageBW$create(
    fn = fn
  )

}



#' @title TensorPoint_create
#'
#' @description Delegates (`__call__`,`decode`,`setup`) to (`encodes`,`decodes`,`setups`) if `split_idx` matches
#'
#'
#' @param ... arguments to pass
#' @return None
#' @export
TensorPoint_create <- function( ...) {

  args = list(
    ...
  )

  if(length(args)==0) {
    invisible(vision()$all$TensorPoint$create)
  } else {

    if(!is.null(args$split_idx)) {
      args$split_idx = as.integer(args$split_idx)
    }

    do.call(vision()$all$TensorPoint$create, args)
  }
}


#' @title TensorBBox
#'
#' @description Basic type for a tensor of bounding boxes in an image
#'
#' @param x tensor
#' @return None
#' @export
TensorBBox <- function(x) {

  vision()$all$TensorBBox(
    x = x
  )

}

#' @title LabeledBBox
#'
#' @description Basic type for a list of bounding boxes in an image
#'
#' @param ... parameters to pass
#' @return None
#' @export
LabeledBBox <- function(...) {

  vision()$all$LabeledBBox(
    ...
  )

}

#' @title ToTensor
#'
#' @description Convert item to appropriate tensor class
#'
#'
#' @param enc encoder
#' @param dec decoder
#' @param split_idx int, split by index
#' @param order order
#' @return None
#' @export
ToTensor <- function(enc = NULL, dec = NULL, split_idx = NULL, order = NULL) {

  vision()$all$ToTensor(
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
#' @param funcs functions
#' @param split_idx split by index
#' @return None
#' @export
Pipeline <- function(funcs = NULL, split_idx = NULL) {

  vision()$all$Pipeline(
    funcs = funcs,
    split_idx = split_idx
  )

}

#' @title TensorImageBW
#'
#' @param x tensor
#' @return None
#' @export
TensorImageBW <- function(x) {

  if(missing(x)) {
    invisible(vision()$all$TensorImageBW)
  } else {
    vision()$all$TensorImageBW(
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
#' @param tfms transformations
#' @param tls tls
#' @param n_inp n_inp
#' @param dl_type DL type
#' @param use_list use list
#' @param do_setup do setup
#' @param split_idx split by index
#' @param train_setup train setup
#' @param splits splits
#' @param types types
#' @param verbose verbose
#' @return None
#' @export
Datasets <- function(items = NULL, tfms = NULL, tls = NULL, n_inp = NULL,
                     dl_type = NULL, use_list = NULL, do_setup = TRUE,
                     split_idx = NULL, train_setup = TRUE, splits = NULL,
                     types = NULL, verbose = FALSE) {

  if(!is.null(n_inp)) {
    n_inp = as.integer(n_inp)
  }

  vision()$all$Datasets(
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
#' @param ... parameters to pass
#' @return None
#' @export
Image = function(...) {
  args = list(...)

  do.call(vision()$all$PILImage, args)
}


#' @title TfmdDL
#'
#' @description Transformed `DataLoader`
#'
#'
#' @param dataset dataset
#' @param bs batch size
#' @param shuffle shuffle
#' @param num_workers number of workers
#' @param verbose verbose
#' @param do_setup do setup
#' @param pin_memory pin memory
#' @param timeout timeout
#' @param batch_size batch size
#' @param drop_last drop last
#' @param indexed indexed
#' @param n int, n
#' @param device device
#' @param ... additional arguments to pass
#' @param after_batch after_batch
#' @return None
#' @export
TfmdDL <- function(dataset, bs = 64, shuffle = FALSE, num_workers = NULL,
                   verbose = FALSE, do_setup = TRUE, pin_memory = FALSE,
                   timeout = 0, batch_size = NULL, drop_last = FALSE,
                   indexed = NULL, n = NULL, device = NULL,
                   after_batch = NULL,
                   ...) {

  args = list(
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
    device = device,
    after_batch = after_batch,
    ...
  )

  strings = c('batch_size', 'n', 'device', 'indexed', 'after_batch')

  for(i in 1:length(strings)) {
    if(is.null(args[[strings[i]]]))
      args[[strings[i]]] <- NULL
  }

  if(os()=='windows' & is.null(args$num_workers)) {
    args$num_workers = 0L
  }

  if(os()=='mac' & is.null(args$num_workers)) {
    args$num_workers = 0L
  }

  if(!is.null(args$num_workers)){
    args$num_workers = as.integer(args$num_workers)
  }

  if(!is.null(args$after_batch)) {
    args$after_batch <- unlist(args$after_batch)
  }

  do.call(vision()$all$TfmdDL, args)

}

#' @title PointScaler
#'
#' @description Scale a tensor representing points
#'
#' @param do_scale do scale
#' @param y_first y first
#' @return None
#' @export
PointScaler <- function(do_scale = TRUE, y_first = FALSE) {

  vision()$all$PointScaler(
    do_scale = do_scale,
    y_first = y_first
  )

}


#' @title Tensor
#'
#' @description Like `torch()$as_tensor`, but handle lists too, and can pass multiple vector elements directly.
#'
#'
#' @param ... image
#' @return None
#' @export
tensor <- function(...) {

  vision()$all$tensor(
    ...
  )

}


#' @title TensorPoint
#'
#' @description Basic type for points in an image
#'
#' @param x tensor
#' @return None
#' @export
TensorPoint <- function(x) {

  vision()$all$TensorPoint(
    x = x
  )

}


#' @title TensorBBox_create
#'
#' @param x tensor
#' @param img_size image size
#' @return None
#' @export
TensorBBox_create <- function(x, img_size = NULL) {

  if(!is.null(img_size)) {
    if(is.list(img_size)) {
      img_size = as.list(as.integer(unlist(img_size)))
    } else {
      img_size = as.integer(img_size)
    }
  }

  vision()$all$TensorBBox$create(
    x = x,
    img_size = img_size
  )

}


#' @title BBoxLabeler
#'
#' @description Delegates (`__call__`,`decode`,`setup`) to (`encodes`,`decodes`,`setups`) if `split_idx` matches
#'
#'
#' @param enc encoder
#' @param dec decoder
#' @param split_idx split by index
#' @param order order
#' @return None
#' @export
BBoxLabeler <- function(enc = NULL, dec = NULL, split_idx = NULL, order = NULL) {

  vision()$all$BBoxLabeler(
    enc = enc,
    dec = dec,
    split_idx = split_idx,
    order = order
  )

}


#' @title TensorMultiCategory
#'
#' @param x tensor
#' @return None
#' @export
TensorMultiCategory <- function(x) {

  vision()$all$TensorMultiCategory(
    x = x
  )

}




#' @title L
#'
#' @description Behaves like a list of `items` but can also index with list of indices or masks
#'
#'
#' @param ... arguments to pass
#'
#' @export
L <- function(...) {

  args = list(...)

  do.call(fastai2$vision$all$L, args)

}


#' @title Params
#'
#' @description Return all parameters of `m`
#'
#'
#' @param m parameters
#' @return None
#' @export
params <- function(m) {

  if(missing(m)) {
    fastai2$vision$all$params
  } else {
    fastai2$vision$all$params(
      m = m
    )
  }

}







