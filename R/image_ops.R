
#' @title Resize
#'
#' @description A transform that before_call its state at each `__call__`
#'
#'
#' @param size size
#' @param method method
#' @param pad_mode pad_mode
#' @param resamples resamples
#'
#' @export
Resize <- function(size, method = "crop", pad_mode = "reflection", resamples = list(2, 0)) {

  args <- list(
    size = as.integer(size),
    method = method,
    pad_mode = pad_mode,
    resamples = as.list(as.integer(unlist(resamples)))
  )

  do.call(vision$all$Resize, args)

}



#' @title Aug_transforms
#'
#' @description Utility func to easily create a list of flip, rotate, zoom, warp, lighting transforms.
#'
#'
#' @param mult mult
#' @param do_flip do_flip
#' @param flip_vert flip_vert
#' @param max_rotate max_rotate
#' @param min_zoom min_zoom
#' @param max_zoom max_zoom
#' @param max_lighting max_lighting
#' @param max_warp max_warp
#' @param p_affine p_affine
#' @param p_lighting p_lighting
#' @param xtra_tfms xtra_tfms
#' @param size size
#' @param mode mode
#' @param pad_mode pad_mode
#' @param align_corners align_corners
#' @param batch batch
#' @param min_scale min_scale
#' @export
aug_transforms <- function(mult = 1.0, do_flip = TRUE, flip_vert = FALSE,
                           max_rotate = 10.0, min_zoom = 1.0, max_zoom = 1.1,
                           max_lighting = 0.2, max_warp = 0.2, p_affine = 0.75,
                           p_lighting = 0.75, xtra_tfms = NULL, size = NULL,
                           mode = "bilinear", pad_mode = "reflection",
                           align_corners = TRUE, batch = FALSE, min_scale = 1.0) {

  args <- list(
    mult = mult,
    do_flip = do_flip,
    flip_vert = flip_vert,
    max_rotate = max_rotate,
    min_zoom = min_zoom,
    max_zoom = max_zoom,
    max_lighting = max_lighting,
    max_warp = max_warp,
    p_affine = p_affine,
    p_lighting = p_lighting,
    xtra_tfms = xtra_tfms,
    size = size,
    mode = mode,
    pad_mode = pad_mode,
    align_corners = align_corners,
    batch = batch,
    min_scale = min_scale
  )

  if(!is.null(size)) {
    args$size = as.integer(args$size)
  }

  aug = do.call(vision$all$aug_transforms, args)

  if(length(aug)==2) {
    return(list(aug[[1]],aug[[2]]))
  } else {
    return(list(aug[[1]],aug[[2]],aug[[3]]))
  }
}


#' @title Imagenet_stats
#'
#' @description list() -> empty list
#'
#' @details list(iterable) -> list initialized from iterable's items If the argument is a list, the return value is the same object.
#'
#'
#' @export
imagenet_stats = function() {
  res = vision$all$imagenet_stats
  return(list(res[[1]],res[[2]]))
}



#' @title Normalize_from_stats
#'
#' @param mean mean
#' @param std std
#' @param dim dim
#' @param ndim ndim
#' @param cuda cuda
#'
#' @export
Normalize_from_stats <- function(mean, std, dim = 1, ndim = 4, cuda = TRUE) {

  if(is.list(mean)) {
    obj = mean
    mean = obj[[1]]
    std = obj[[2]]
    args <- list(
      mean = mean,
      std = std,
      dim = as.integer(dim),
      ndim = as.integer(ndim),
      cuda = cuda
    )
  } else {
    args <- list(
      mean = mean,
      std = std,
      dim = as.integer(dim),
      ndim = as.integer(ndim),
      cuda = cuda
    )
  }

  do.call(vision$all$Normalize$from_stats, args)

}



#' @title Rotate
#'
#' @description Apply a random rotation of at most `max_deg` with probability `p` to a batch of images
#'
#' @details
#'
#' @param max_deg max_deg
#' @param p p
#' @param draw draw
#' @param size size
#' @param mode mode
#' @param pad_mode pad_mode
#' @param align_corners align_corners
#' @param batch batch
#'
#' @export
Rotate <- function(max_deg = 10, p = 0.5, draw = NULL, size = NULL,
                   mode = "bilinear", pad_mode = "reflection",
                   align_corners = TRUE, batch = FALSE) {

  vision$all$Rotate(
    max_deg = as.integer(max_deg),
    p = p,
    draw = draw,
    size = size,
    mode = mode,
    pad_mode = pad_mode,
    align_corners = align_corners,
    batch = batch
  )

}

#' @title Flip
#'
#' @description Randomly flip a batch of images with a probability `p`
#'
#'
#' @param p p
#' @param draw draw
#' @param size size
#' @param mode mode
#' @param pad_mode pad_mode
#' @param align_corners align_corners
#' @param batch batch
#'
#' @export
Flip <- function(p = 0.5, draw = NULL, size = NULL, mode = "bilinear",
                 pad_mode = "reflection", align_corners = TRUE,
                 batch = FALSE) {

  vision$all$Flip(
    p = p,
    draw = draw,
    size = size,
    mode = mode,
    pad_mode = pad_mode,
    align_corners = align_corners,
    batch = batch
  )

}


#' @title Dihedral
#'
#' @description Apply a random dihedral transformation to a batch of images with a probability `p`
#'
#'
#' @param p p
#' @param draw draw
#' @param size size
#' @param mode mode
#' @param pad_mode pad_mode
#' @param align_corners align_corners
#' @param batch batch
#'
#' @export
Dihedral <- function(p = 0.5, draw = NULL, size = NULL,
                     mode = "bilinear", pad_mode = "reflection",
                     align_corners = NULL, batch = FALSE) {

  vision$all$Dihedral(
    p = p,
    draw = draw,
    size = size,
    mode = mode,
    pad_mode = pad_mode,
    align_corners = align_corners,
    batch = batch
  )

}









