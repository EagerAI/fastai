
#' @title RandTransform
#'
#' @description A transform that before_call its state at each `__call__`
#'
#'
#' @param p probability
#' @param nm nm
#' @param before_call before call
#' @param ... additional arguments to pass
#' @return None
#' @export
RandTransform <- function(p = 1.0, nm = NULL, before_call = NULL,
                          ...) {

  vision$all$RandTransform(
    p = p,
    nm = nm,
    before_call = before_call,
    ...
  )

}


#' @title FlipItem
#'
#' @description Randomly flip with probability `p`
#'
#'
#' @param p probability
#' @return None
#' @export
FlipItem <- function(p = 0.5) {

  vision$all$FlipItem(
    p = p
  )

}


#' @title DihedralItem
#'
#' @description Randomly flip with probability `p`
#'
#'
#' @param p probability
#' @param nm nm
#' @param before_call before call
#' @return None
#' @export
DihedralItem <- function(p = 1.0, nm = NULL, before_call = NULL) {

  vision$all$DihedralItem(
    p = p,
    nm = nm,
    before_call = before_call
  )

}

#' @title CropPad
#'
#' @description Center crop or pad an image to `size`
#'
#' @return None
#' @param size size
#' @param pad_mode padding mode
#' @param ... additional arguments
#' @export
CropPad <- function(size, pad_mode = "zeros",
                    ...) {

  vision$all$CropPad(
    size = size,
    pad_mode = pad_mode,
    ...
  )

}

#' @title RandomCrop
#'
#' @description Randomly crop an image to `size`
#'
#' @param ... additional arguments
#' @param size size
#' @return None
#' @export
RandomCrop <- function(size, ...) {

  vision$all$RandomCrop(
    size = as.integer(size),
    ...
  )

}


#' @title OldRandomCrop
#'
#' @description Randomly crop an image to `size`
#' @param ... additional arguments
#' @return None
#'
#' @param size size
#' @param pad_mode padding mode
#'
#' @export
OldRandomCrop <- function(size, pad_mode = "zeros", ...) {

  vision$all$OldRandomCrop(
    size = size,
    pad_mode = pad_mode,
    ...
  )

}


#' @title RandomResizedCrop
#'
#' @description Picks a random scaled crop of an image and resize it to `size`
#'
#' @param size size
#' @param min_scale minimum scale
#' @param ratio ratio
#' @param resamples resamples
#' @param val_xtra validation xtra
#' @return None
#' @export
RandomResizedCrop <- function(size, min_scale = 0.08, ratio = list(0.75, 1.3333333333333333),
                              resamples = list(2, 0), val_xtra = 0.14) {

  vision$all$RandomResizedCrop(
    size = as.integer(size),
    min_scale = min_scale,
    ratio = ratio,
    resamples = as.list(as.integer(unlist(resamples))),
    val_xtra = val_xtra
  )

}

#' @title RatioResize
#'
#' @description Resizes the biggest dimension of an image to `max_sz` maintaining the aspect ratio
#'
#'
#' @param max_sz maximum sz
#' @param resamples resamples
#' @param ... additional arguments
#' @return None
#' @export
RatioResize <- function(max_sz, resamples = list(2, 0),
                        ...) {

  vision$all$RatioResize(
    max_sz = max_sz,
    resamples = as.list(as.integer(unlist(resamples))),
    ...
  )

}



#' @title TensorImage
#'
#'
#' @param x tensor
#' @return None
#' @export
TensorImage <- function(x) {

  vision$all$TensorImage(
    x = x
  )

}

#' @title Aaffine_coord
#'
#'
#' @param x tensor
#' @param mat mat
#' @param coord_tfm coordinate tfm
#' @param sz sz
#' @param mode mode
#' @param pad_mode padding mode
#' @param align_corners align corners
#' @param ... additional arguments
#' @return None
#' @export
affine_coord <- function(x, mat = NULL, coord_tfm = NULL, sz = NULL,
                         mode = "bilinear", pad_mode = "reflection",
                         align_corners = TRUE, ...) {

  vision$all$TensorImage$affine_coord(
    x = x,
    mat = mat,
    coord_tfm = coord_tfm,
    sz = sz,
    mode = mode,
    pad_mode = pad_mode,
    align_corners = align_corners,
    ...
  )

}


#' @title AffineCoordTfm
#'
#' @description Combine and apply affine and coord transforms
#'
#'
#' @param aff_fs aff fs
#' @param coord_fs coordinate fs
#' @param size size
#' @param mode mode
#' @param pad_mode padding mode
#' @param mode_mask mode mask
#' @param align_corners align corners
#' @return None
#' @export
AffineCoordTfm <- function(aff_fs = NULL, coord_fs = NULL, size = NULL,
                           mode = "bilinear", pad_mode = "reflection",
                           mode_mask = "nearest", align_corners = NULL) {

  vision$all$AffineCoordTfm(
    aff_fs = aff_fs,
    coord_fs = coord_fs,
    size = size,
    mode = mode,
    pad_mode = pad_mode,
    mode_mask = mode_mask,
    align_corners = align_corners
  )

}

#' @title RandomResizedCropGPU
#'
#' @description Picks a random scaled crop of an image and resize it to `size`
#'
#'
#' @param size size
#' @param min_scale minimum scale
#' @param ratio ratio
#' @param mode mode
#' @param valid_scale validation scale
#' @return None
#' @export
RandomResizedCropGPU <- function(size, min_scale = 0.08, ratio = list(0.75, 1.3333333333333333),
                                 mode = "bilinear", valid_scale = 1.0) {

  vision$all$RandomResizedCropGPU(
    size = size,
    min_scale = min_scale,
    ratio = ratio,
    mode = mode,
    valid_scale = valid_scale
  )

}


#' @title Affline mat
#'
#' @param ... parameters to pass
#' @return None
#' @export
affine_mat = function(...) {
  vision$all$affine_mat(...)
}



#' @title Mask_tensor
#'
#' @description Mask elements of `x` with `neutral` with probability `1-p`
#'
#'
#' @param x tensor
#' @param p probability
#' @param neutral neutral
#' @param batch batch
#' @return None
#' @export
mask_tensor <- function(x, p = 0.5, neutral = 0.0, batch = FALSE) {

  vision$all$mask_tensor(
    x = x,
    p = p,
    neutral = neutral,
    batch = batch
  )

}


#' @title Flip_mat
#'
#' @description Return a random flip matrix
#'
#'
#' @param x tensor
#' @param p probability
#' @param draw draw
#' @param batch batch
#' @return None
#' @export
flip_mat <- function(x, p = 0.5, draw = NULL, batch = FALSE) {

  vision$all$flip_mat(
    x = x,
    p = p,
    draw = draw,
    batch = batch
  )

}


#' @title DeterministicDraw
#' @param vals values
#' @return None
#' @export
DeterministicDraw <- function(vals) {

  vision$all$DeterministicDraw(
    vals = vals
  )

}

#' @title DeterministicFlip
#'
#' @description Flip the batch every other call
#'
#'
#' @param size size
#' @param mode mode
#' @param pad_mode padding mode
#' @param align_corners align corners
#' @param ... parameters to pass
#' @return None
#' @export
DeterministicFlip <- function(size = NULL, mode = "bilinear",
                              pad_mode = "reflection", align_corners = TRUE,
                              ...) {

  vision$all$DeterministicFlip(
    size = size,
    mode = mode,
    pad_mode = pad_mode,
    align_corners = align_corners,
    ...
  )

}

#' @title Dihedral_mat
#'
#' @description Return a random dihedral matrix
#'
#' @param x tensor
#' @param p probability
#' @param draw draw
#' @param batch batch
#' @return None
#' @export
dihedral_mat <- function(x, p = 0.5, draw = NULL, batch = FALSE) {

  vision$all$dihedral_mat(
    x = x,
    p = p,
    draw = draw,
    batch = batch
  )

}


#' @title Dihedral
#'
#' @description Apply a random dihedral transformation to a batch of images with a probability `p`
#'
#'
#' @param p probability
#' @param draw draw
#' @param size size
#' @param mode mode
#' @param pad_mode padding mode
#' @param align_corners align corners
#' @param batch batch
#' @return None
#' @export
Dihedral <- function(p = 0.5, draw = NULL, size = NULL, mode = "bilinear",
                     pad_mode = "reflection", align_corners = NULL, batch = FALSE) {

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

#' @title DeterministicDihedral
#'
#' @description Apply a random dihedral transformation to a batch of images with a probability `p`
#'
#'
#' @param size size
#' @param mode mode
#' @param pad_mode padding mode
#' @param align_corners align corners
#' @return None
#' @export
DeterministicDihedral <- function(size = NULL, mode = "bilinear",
                                  pad_mode = "reflection", align_corners = NULL) {

  vision$all$DeterministicDihedral(
    size = size,
    mode = mode,
    pad_mode = pad_mode,
    align_corners = align_corners
  )

}

#' @title Rotate_mat
#'
#' @description Return a random rotation matrix with `max_deg` and `p`
#'
#'
#' @param x tensor
#' @param max_deg max_deg
#' @param p probability
#' @param draw draw
#' @param batch batch
#' @return None
#' @export
rotate_mat <- function(x, max_deg = 10, p = 0.5, draw = NULL, batch = FALSE) {

  python_function_result <- vision$all$rotate_mat(
    x = x,
    max_deg = as.integer(max_deg),
    p = p,
    draw = draw,
    batch = batch
  )

}


#' @title Zoom_mat
#'
#' @description Return a random zoom matrix with `max_zoom` and `p`
#'
#'
#' @param x tensor
#' @param min_zoom minimum zoom
#' @param max_zoom maximum zoom
#' @param p probability
#' @param draw draw
#' @param draw_x draw x
#' @param draw_y draw y
#' @param batch batch
#' @return None
#' @export
zoom_mat <- function(x, min_zoom = 1.0, max_zoom = 1.1, p = 0.5, draw = NULL,
                     draw_x = NULL, draw_y = NULL, batch = FALSE) {

  vision$all$zoom_mat(
    x = x,
    min_zoom = min_zoom,
    max_zoom = max_zoom,
    p = p,
    draw = draw,
    draw_x = draw_x,
    draw_y = draw_y,
    batch = batch
  )

}


#' @title Find_coeffs
#'
#' @description Find coefficients for warp tfm from `p1` to `p2`
#'
#'
#' @param p1 coefficient p1
#' @param p2 coefficient p2
#' @return None
#' @export
find_coeffs <- function(p1, p2) {

  vision$all$find_coeffs(
    p1 = p1,
    p2 = p2
  )

}


#' @title Apply_perspective
#'
#' @description Apply perspective tranfom on `coords` with `coeffs`
#'
#'
#' @param coords coordinates
#' @param coeffs coefficient
#' @return None
#' @export
apply_perspective <- function(coords, coeffs) {

  vision$all$apply_perspective(
    coords = coords,
    coeffs = coeffs
  )

}


#' @title Warp
#'
#' @description Apply perspective warping with `magnitude` and `p` on a batch of matrices
#'
#'
#' @param magnitude magnitude
#' @param p probability
#' @param draw_x draw x
#' @param draw_y draw y
#' @param size size
#' @param mode mode
#' @param pad_mode padding mode
#' @param batch batch
#' @param align_corners align corners
#' @return None
#' @export
Warp <- function(magnitude = 0.2, p = 0.5, draw_x = NULL, draw_y = NULL,
                 size = NULL, mode = "bilinear", pad_mode = "reflection",
                 batch = FALSE, align_corners = TRUE) {

  vision$all$Warp(
    magnitude = magnitude,
    p = p,
    draw_x = draw_x,
    draw_y = draw_y,
    size = size,
    mode = mode,
    pad_mode = pad_mode,
    batch = batch,
    align_corners = align_corners
  )

}


#' @title LightingTfm
#'
#' @description Apply `fs` to the logits
#'
#'
#' @param fs fs
#' @param ... parameters to pass
#' @return None
#' @export
LightingTfm <- function(fs, ...) {


  vision$all$LightingTfm(
    fs = fs,
    ...
  )

}

#' @title Contrast
#'
#' @description Apply change in contrast of `max_lighting` to batch of images with probability `p`.
#'
#'
#' @param max_lighting maximum lighting
#' @param p probability
#' @param draw draw
#' @param batch batch
#' @return None
#' @export
Contrast <- function(max_lighting = 0.2, p = 0.75, draw = NULL, batch = FALSE) {

  vision$all$Contrast(
    max_lighting = max_lighting,
    p = p,
    draw = draw,
    batch = batch
  )

}

#' @title Grayscale
#'
#' @description Tensor to grayscale tensor. Uses the ITU-R 601-2 luma transform.
#'
#'
#' @param x tensor
#' @return None
#' @export
grayscale <- function(x) {

  vision$all$grayscale(
    x = x
  )

}


#' @title Saturation
#'
#' @description Apply change in saturation of `max_lighting` to batch of images with probability `p`.
#'
#'
#' @param max_lighting maximum lighting
#' @param p probability
#' @param draw draw
#' @param batch batch
#' @return None
#' @export
Saturation <- function(max_lighting = 0.2, p = 0.75, draw = NULL, batch = FALSE) {

  vision$all$Saturation(
    max_lighting = max_lighting,
    p = p,
    draw = draw,
    batch = batch
  )

}

#' @title Rgb2hsv
#'
#' @description Converts a RGB image to an HSV image.
#'
#' @details Note: Will not work on logit space images.
#'
#' @param img image object
#' @return None
#' @export
rgb2hsv <- function(img) {

  vision$all$rgb2hsv(
    img = img
  )

}

#' @title Hsv2rgb
#'
#' @description Converts a HSV image to an RGB image.
#'
#'
#' @param img image object
#' @return None
#' @export
hsv2rgb <- function(img) {

  vision$all$hsv2rgb(
    img = img
  )

}


#' @title Hue
#'
#' @description Apply change in hue of `max_hue` to batch of images with probability `p`.
#'
#' @param max_hue maximum hue
#' @param p probability
#' @param draw draw
#' @param batch batch
#' @return None
#' @export
Hue <- function(max_hue = 0.1, p = 0.75, draw = NULL, batch = FALSE) {

  vision$all$Hue(
    max_hue = max_hue,
    p = p,
    draw = draw,
    batch = batch
  )

}


#' @title RandomErasing
#'
#' @description Randomly selects a rectangle region in an image and randomizes its pixels.
#'
#'
#' @param p probability
#' @param sl sl
#' @param sh sh
#' @param min_aspect minimum aspect
#' @param max_count maximum count
#' @return None
#' @export
RandomErasing <- function(p = 0.5, sl = 0.0, sh = 0.3, min_aspect = 0.3, max_count = 1) {

  vision$all$RandomErasing(
    p = p,
    sl = sl,
    sh = sh,
    min_aspect = min_aspect,
    max_count = as.integer(max_count)
  )

}

#' @title Cutout_gaussian
#'
#' @description Replace all `areas` in `x` with N(0,1) noise
#'
#'
#' @param x tensor
#' @param areas areas
#' @return None
#' @export
cutout_gaussian <- function(x, areas) {

  vision$all$cutout_gaussian(
    x = x,
    areas = areas
  )

}


#' @title Norm_apply_denorm
#'
#' @description Normalize `x` with `nrm`, then apply `f`, then denormalize
#'
#'
#' @param x tensor
#' @param f function
#' @param nrm nrm
#' @return None
#' @export
norm_apply_denorm <- function(x, f, nrm) {

  vision$all$norm_apply_denorm(
    x = x,
    f = f,
    nrm = nrm
  )

}



#' @title Setup_aug_tfms
#'
#' @description Go through `tfms` and combines together affine/coord or lighting transforms
#'
#'
#' @param tfms transformations
#' @return None
#' @export
setup_aug_tfms <- function(tfms) {

  vision$all$setup_aug_tfms(
    tfms = tfms
  )

}


#' @title Get_annotations
#'
#' @description Open a COCO style json in `fname` and returns the lists of filenames (with maybe `prefix`) and labelled bboxes.
#'
#'
#' @param fname folder name
#' @param prefix prefix
#' @return None
#' @export
get_annotations <- function(fname, prefix = NULL) {

  vision$all$get_annotations(
    fname = fname,
    prefix = prefix
  )

}


#' @title MultiCategorize
#'
#' @description Reversible transform of multi-category strings to `vocab` id
#'
#'
#' @param vocab vocabulary
#' @param add_na add NA
#' @return None
#' @export
MultiCategorize <- function(vocab = NULL, add_na = FALSE) {

  vision$all$MultiCategorize(
    vocab = vocab,
    add_na = add_na
  )

}

#' @title IntToFloatTensor
#'
#' @description Transform image to float tensor, optionally dividing by 255 (e.g. for images).
#'
#'
#' @param div divide value
#' @param div_mask divide mask
#' @return None
#' @export
IntToFloatTensor <- function(div = 255.0, div_mask = 1) {

  vision$all$IntToFloatTensor(
    div = div,
    div_mask = as.integer(div_mask)
  )

}


#' @title FuncSplitter
#'
#' @description Split `items` by result of `func` (`TRUE` for validation, `FALSE` for training set).
#'
#'
#' @param func function
#' @return None
#' @export
FuncSplitter <- function(func) {

  vision$all$FuncSplitter(
    func = func
  )

}



#' @title show_image
#'
#' @description Show a PIL or PyTorch image on `ax`.
#'
#'
#' @param im im
#' @param ax axis
#' @param figsize figure size
#' @param title title
#' @param ctx ctx
#' @param cmap color maps
#' @param norm normalization
#' @param aspect aspect
#' @param interpolation interpolation
#' @param alpha alpha value
#' @param vmin value min
#' @param vmax value max
#' @param origin origin
#' @param extent extent
#'
#' @export
show_image <- function(im, ax = NULL, figsize = NULL, title = NULL, ctx = NULL,
                       cmap = NULL, norm = NULL, aspect = NULL, interpolation = NULL,
                       alpha = NULL, vmin = NULL, vmax = NULL, origin = NULL, extent = NULL) {

  args <- list(
    im = im,
    ax = ax,
    figsize = figsize,
    title = title,
    ctx = ctx,
    cmap = cmap,
    norm = norm,
    aspect = aspect,
    interpolation = interpolation,
    alpha = alpha,
    vmin = vmin,
    vmax = vmax,
    origin = origin,
    extent = extent
  )

  do.call(vision$all$show_image, args)

}












