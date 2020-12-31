

#' @title Aug_tfms
#'
#' @description Collection of useful augmentation transforms.
#'
#' @param size The final size of the image. If an `int` is given, the maximum size of the image is rescaled, maintaing aspect ratio. If a `list` is given, the image is rescaled to have that exact size (height, width).
#' @param presize presize
#' @param horizontal_flip Flip around the y-axis. If `NULL` this transform is not applied.
#' @param shift_scale_rotate Randomly shift, scale, and rotate. If `NULL` this transform is not applied.
#' @param rgb_shift Randomly shift values for each channel of RGB image. If `NULL` this transform is not applied.
#' @param lightning Randomly changes Brightness and Contrast. If `NULL` this transform is not applied.
#' @param blur Randomly blur the image. If `NULL` this transform is not applied.
#' @param crop_fn Randomly crop the image. If `NULL` this transform is not applied. Use `partial` to saturate other parameters of the class.
#' @param pad Pad the image to `size`, squaring the image if `size` is an `int`. If `NULL` this transform is not applied. Use `partial` to sature other parameters of the class.
#' @return None
#' @export
icevision_aug_tfms <- function(size, presize = NULL,
                               horizontal_flip = icevision_HorizontalFlip(always_apply=FALSE, p=0.5),
                               shift_scale_rotate = icevision_ShiftScaleRotate(always_apply=FALSE, p=0.5,
                                                                               shift_limit_x=c(-0.0625, 0.0625),
                                                                               shift_limit_y=c(-0.0625, 0.0625),
                                                                               scale_limit=c(-0.09999999999999998,
                                                                                             0.10000000000000009),
                                                                               rotate_limit=c(-45, 45), interpolation=1,
                                                                               border_mode=4, value=NULL, mask_value=NULL),
                               rgb_shift = icevision_RGBShift(always_apply=FALSE, p=0.5, r_shift_limit=c(-20, 20),
                                                              g_shift_limit=c(-20, 20), b_shift_limit=c(-20, 20)),
                               lightning = icevision_RandomBrightnessContrast(always_apply=FALSE, p=0.5,
                                                                              brightness_limit=c(-0.2, 0.2),
                                                                              contrast_limit=c(-0.2, 0.2),
                                                                              brightness_by_max=TRUE),
                               blur = icevision_Blur(always_apply=FALSE, p=0.5, blur_limit=c(1, 3)),
                               crop_fn = partial(icevision_RandomSizedBBoxSafeCrop, p=0.5),
                               pad = partial(icevision_PadIfNeeded, border_mode=0, value=list(124, 116, 104))) {

  args <- list(
    size = as.integer(size),
    presize = presize,
    horizontal_flip = horizontal_flip,
    shift_scale_rotate = shift_scale_rotate,
    rgb_shift = rgb_shift,
    lightning = lightning,
    blur = blur,
    crop_fn = crop_fn,
    pad = pad
  )

  if(!is.null(args$presize))
    args$presize <- as.integer(args$presize)

  do.call(icevision()$tfms$albumentations$aug_tfms, args)

}

#' @title HorizontalFlip
#'
#' @description Flip the input horizontally around the y-axis.
#'
#'
#' @param always_apply always_apply
#' @param p p
#'
#' @section Targets:
#' image, mask, bboxes, keypoints
#'
#' @section Image types:
#' uint8, float32
#' @return None
#' @export
icevision_HorizontalFlip <- function(always_apply = FALSE, p = 0.5) {

  icevision()$tfms$albumentations$transforms$HorizontalFlip(
    always_apply = always_apply,
    p = p
  )

}


#' @title ShiftScaleRotate
#'
#' @description Randomly apply affine transforms: translate, scale and rotate the input.
#'
#'
#' @param shift_limit shift_limit
#' @param scale_limit scale_limit
#' @param rotate_limit rotate_limit
#' @param interpolation interpolation
#' @param border_mode border_mode
#' @param value value
#' @param mask_value mask_value
#' @param shift_limit_x shift_limit_x
#' @param shift_limit_y shift_limit_y
#' @param always_apply always_apply
#' @param p p
#'
#' @section Targets:
#' image, mask, keypoints
#'
#' @section Image types:
#' uint8, float32
#' @return None
#' @export
icevision_ShiftScaleRotate <- function(shift_limit = 0.0625, scale_limit = 0.1, rotate_limit = 45,
                                       interpolation = 1, border_mode = 4, value = NULL, mask_value = NULL,
                                       shift_limit_x = NULL, shift_limit_y = NULL, always_apply = FALSE, p = 0.5) {

  args = list(
    shift_limit = shift_limit,
    scale_limit = scale_limit,
    rotate_limit = as.integer(rotate_limit),
    interpolation = as.integer(interpolation),
    border_mode = as.integer(border_mode),
    value = value,
    mask_value = mask_value,
    shift_limit_x = shift_limit_x,
    shift_limit_y = shift_limit_y,
    always_apply = always_apply,
    p = p
  )


  if(is.null(args$value))
    args$value <- NULL

  if(is.null(args$mask_value))
    mask_value <- NULL

  if(is.null(args$shift_limit_x))
    args$shift_limit_x <- NULL

  if(is.null(args$shift_limit_y))
    args$shift_limit_y <- NULL

  do.call(icevision()$tfms$albumentations$transforms$ShiftScaleRotate, args)

}



#' @title RGBShift
#'
#' @description Randomly shift values for each channel of the input RGB image.
#'
#'
#' @param r_shift_limit r_shift_limit
#' @param g_shift_limit g_shift_limit
#' @param b_shift_limit b_shift_limit
#' @param always_apply always_apply
#' @param p p
#'
#' @section Targets:
#' image
#'
#' @section Image types:
#' uint8, float32
#' @return None
#' @export
icevision_RGBShift <- function(r_shift_limit = 20, g_shift_limit = 20, b_shift_limit = 20,
                               always_apply = FALSE, p = 0.5) {

  args <- list(
    r_shift_limit = as.integer(r_shift_limit),
    g_shift_limit = as.integer(g_shift_limit),
    b_shift_limit = as.integer(b_shift_limit),
    always_apply = always_apply,
    p = p
  )


  do.call(icevision()$tfms$albumentations$transforms$RGBShift, args)

}


#' @title RandomBrightnessContrast
#'
#' @description Randomly change brightness and contrast of the input image.
#'
#'
#' @param brightness_limit brightness_limit
#' @param contrast_limit contrast_limit
#' @param brightness_by_max brightness_by_max
#' @param always_apply always_apply
#' @param p p
#'
#' @section Targets:
#' image
#'
#' @section Image types:
#' uint8, float32
#' @return None
#' @export
icevision_RandomBrightnessContrast <- function(brightness_limit = 0.2, contrast_limit = 0.2,
                                               brightness_by_max = TRUE, always_apply = FALSE,
                                               p = 0.5) {

  args <- list(
    brightness_limit = brightness_limit,
    contrast_limit = contrast_limit,
    brightness_by_max = brightness_by_max,
    always_apply = always_apply,
    p = p
  )


  do.call(icevision()$tfms$albumentations$transforms$RandomBrightnessContrast, args)

}


#' @title Blur
#'
#' @description Blur the input image using a random-sized kernel.
#'
#'
#' @param blur_limit blur_limit
#' @param always_apply always_apply
#' @param p p
#'
#' @section Targets:
#' image
#'
#' @section Image types:
#' uint8, float32
#' @return None
#' @export
icevision_Blur <- function(blur_limit = 7, always_apply = FALSE, p = 0.5) {

  args <- list(
    blur_limit = as.integer(blur_limit),
    always_apply = always_apply,
    p = p
  )

  do.call(icevision()$tfms$albumentations$transforms$Blur, args)

}


#' @title DualTransform
#'
#' @description Transform for segmentation task.
#'
#'
#' @param always_apply always_apply
#' @param p p
#' @return None
#' @export
icevision_DualTransform <- function(always_apply = FALSE, p = 0.5) {

  icevision()$tfms$albumentations$core$transforms_interface$DualTransform(
    always_apply = always_apply,
    p = p
  )

}



#' @title RandomSizedBBoxSafeCrop
#'
#' @description Crop a random part of the input and rescale it to some size without loss of bboxes.
#'
#'
#' @param height height
#' @param width width
#' @param erosion_rate erosion_rate
#' @param interpolation interpolation
#' @param always_apply always_apply
#' @param p p
#'
#' @section Targets:
#' image, mask, bboxes
#'
#' @section Image types:
#' uint8, float32
#' @return None
#' @export
icevision_RandomSizedBBoxSafeCrop <- function(height, width, erosion_rate = 0.0, interpolation = 1,
                                              always_apply = FALSE, p = 1.0) {

  if(missing(height) & missing(width)) {
    icevision()$tfms$albumentations$transforms$RandomSizedBBoxSafeCrop
  } else {
    args <- list(
      height = as.integer(height),
      width = as.integer(width),
      erosion_rate = erosion_rate,
      interpolation = as.integer(interpolation),
      always_apply = always_apply,
      p = p
    )


    do.call(icevision()$tfms$albumentations$transforms$RandomSizedBBoxSafeCrop, args)
  }

}



#' @title PadIfNeeded
#'
#' @description Pad side of the image / max if side is less than desired number.
#'
#' @param min_height min_height
#' @param min_width min_width
#' @param pad_height_divisor pad_height_divisor
#' @param pad_width_divisor pad_width_divisor
#' @param border_mode border_mode
#' @param value value
#' @param mask_value mask_value
#' @param always_apply always_apply
#' @param p p
#'
#' @section Targets:
#' image, mask, bbox, keypoints
#'
#' @section Image types:
#' uint8, float32
#'
#' @export
icevision_PadIfNeeded <- function(min_height = 1024, min_width = 1024, pad_height_divisor = NULL,
                                  pad_width_divisor = NULL, border_mode = 4, value = NULL,
                                  mask_value = NULL, always_apply = FALSE, p = 1.0) {

  args <- list(
    min_height = as.integer(min_height),
    min_width = as.integer(min_width),
    pad_height_divisor = pad_height_divisor,
    pad_width_divisor = pad_width_divisor,
    border_mode = as.integer(border_mode),
    value = value,
    mask_value = mask_value,
    always_apply = always_apply,
    p = p
  )


  if(is.null(args$pad_height_divisor))
    args$pad_height_divisor <- NULL

  if(is.null(args$pad_width_divisor))
    args$pad_width_divisor <- NULL

  if(is.null(args$value))
    args$value <- NULL
  else
    args$value <- as.list(as.integer(unlist(args$value)))

  if(is.null(args$mask_value))
    args$mask_value <- NULL


  do.call(icevision()$tfms$albumentations$transforms$PadIfNeeded, args)

}


#' @title Adapter
#'
#' @description Adapter that enables the use of albumentations transforms.
#'
#'
#' @param tfms `Sequence` of albumentation transforms.
#' @return None
#' @export
icevision_Adapter <- function(tfms) {

  icevision()$tfms$albumentations$Adapter(
    tfms = tfms
  )

}



#' @title HueSaturationValue
#'
#' @description Randomly change hue, saturation and value of the input image.
#'
#'
#' @param hue_shift_limit hue_shift_limit
#' @param sat_shift_limit sat_shift_limit
#' @param val_shift_limit val_shift_limit
#' @param always_apply always_apply
#' @param p p
#'
#' @section Targets:
#' image
#'
#' @section Image types:
#' uint8, float32
#' @return None
#' @export
icevision_HueSaturationValue <- function(hue_shift_limit = 20,
                                         sat_shift_limit = 30,
                                         val_shift_limit = 20,
                                         always_apply = FALSE, p = 0.5) {

  icevision()$tfms$albumentations$transforms$HueSaturationValue(
    hue_shift_limit = as.integer(hue_shift_limit),
    sat_shift_limit = as.integer(sat_shift_limit),
    val_shift_limit = as.integer(val_shift_limit),
    always_apply = always_apply,
    p = p
  )

}

#' @title BasicIAATransform
#'
#'
#' @param always_apply always_apply
#' @param p p
#' @return None
#' @export
icevision_BasicIAATransform <- function(always_apply = FALSE, p = 0.5) {

  icevision()$tfms$albumentations$BasicIAATransform(
    always_apply = always_apply,
    p = p
  )

}


#' @title BasicTransform
#'
#'
#' @param always_apply always_apply
#' @param p p
#' @return None
#' @export
icevision_BasicTransform <- function(always_apply = FALSE, p = 0.5) {

  icevision()$tfms$albumentations$BasicTransform(
    always_apply = always_apply,
    p = p
  )

}



#' @title ChannelDropout
#'
#' @description Randomly Drop Channels in the input Image.
#'
#'
#' @param channel_drop_range channel_drop_range
#' @param fill_value fill_value
#' @param always_apply always_apply
#' @param p p
#'
#' @section Targets:
#' image
#'
#' @section Image types:
#' uint8, uint16, unit32, float32
#'
#' @export
icevision_ChannelDropout <- function(channel_drop_range = list(1, 1), fill_value = 0, always_apply = FALSE, p = 0.5) {

  icevision()$tfms$albumentations$ChannelDropout(
    channel_drop_range = as.list(as.integer(unlist(channel_drop_range))),
    fill_value = as.integer(fill_value),
    always_apply = always_apply,
    p = p
  )

}

#' @title ChannelShuffle
#'
#' @description Randomly rearrange channels of the input RGB image.
#'
#' @param always_apply always_apply
#' @param p p
#'
#' @section Targets:
#' image
#'
#' @section Image types:
#' uint8, float32
#' @return None
#' @export
icevision_ChannelShuffle <- function(always_apply = FALSE, p = 0.5) {

  icevision()$tfms$albumentations$ChannelShuffle(
    always_apply = always_apply,
    p = p
  )

}


#' @title CoarseDropout
#'
#' @description CoarseDropout of the rectangular regions in the image.
#'
#'
#' @param max_holes max_holes
#' @param max_height max_height
#' @param max_width max_width
#' @param min_holes min_holes
#' @param min_height min_height
#' @param min_width min_width
#' @param fill_value fill_value
#' @param mask_fill_value mask_fill_value
#' @param always_apply always_apply
#' @param p p
#'
#' @section Targets:
#' image, mask
#'
#' @section Image types:
#' uint8, float32
#'
#' @section Reference:
#' | https://arxiv.org/abs/1708.04552 | https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py | https://github.com/aleju/imgaug/blob/master/imgaug/augmenters/arithmetic.py
#' @return None
#' @export
icevision_CoarseDropout <- function(max_holes = 8, max_height = 8,
                                    max_width = 8, min_holes = NULL,
                                    min_height = NULL, min_width = NULL, fill_value = 0,
                                    mask_fill_value = NULL, always_apply = FALSE, p = 0.5) {

  args <- list(
    max_holes = as.integer(max_holes),
    max_height = as.integer(max_height),
    max_width = as.integer(max_width),
    min_holes = min_holes,
    min_height = min_height,
    min_width = min_width,
    fill_value = as.integer(fill_value),
    mask_fill_value = mask_fill_value,
    always_apply = always_apply,
    p = p
  )

  if(is.null(args$min_holes))
    args$min_holes <- NULL
  else
    args$min_holes <- as.integer(args$min_holes)

  if(is.null(args$min_height))
    args$min_height <- NULL
  else
    args$min_height <- as.integer(args$min_height)

  if(is.null(args$min_width))
    args$min_width <- NULL
  else
    args$min_width <- as.integer(args$min_width)

  if(is.null(args$mask_fill_value))
    args$mask_fill_value <- NULL
  else
    args$mask_fill_value <- as.integer(args$mask_fill_value)

  do.call(icevision()$tfms$albumentations$CoarseDropout, args)

}


#' @title ColorJitter
#'
#' @description Randomly changes the brightness, contrast, and saturation of an image. Compared to ColorJitter from torchvision,
#'
#' @details this transform gives a little bit different results because Pillow (used in torchvision) and OpenCV (used in
#' Albumentations) transform an image to HSV format by different formulas. Another difference - Pillow uses uint8
#' overflow, but we use value saturation.
#'
#' @param brightness brightness
#' @param contrast contrast
#' @param saturation saturation
#' @param hue hue
#' @param always_apply always_apply
#' @param p p
#' @return None
#' @export
icevision_ColorJitter <- function(brightness = 0.2, contrast = 0.2, saturation = 0.2,
                                  hue = 0.2, always_apply = FALSE, p = 0.5) {

  icevision()$tfms$albumentations$ColorJitter(
    brightness = brightness,
    contrast = contrast,
    saturation = saturation,
    hue = hue,
    always_apply = always_apply,
    p = p
  )

}

#' @title CLAHE
#'
#' @description Apply Contrast Limited Adaptive Histogram Equalization to the input image.
#'
#'
#' @param clip_limit clip_limit
#' @param tile_grid_size tile_grid_size
#' @param always_apply always_apply
#' @param p p
#'
#' @section Targets:
#' image
#'
#' @section Image types:
#' uint8
#' @return None
#' @export
icevision_CLAHE <- function(clip_limit = 4.0, tile_grid_size = list(8, 8), always_apply = FALSE, p = 0.5) {

  icevision()$tfms$albumentations$CLAHE(
    clip_limit = clip_limit,
    tile_grid_size = as.list(as.integer(unlist(tile_grid_size))),
    always_apply = always_apply,
    p = p
  )

}


#' @title Crop
#'
#' @description Crop region from image.
#'
#'
#' @param x_min x_min
#' @param y_min y_min
#' @param x_max x_max
#' @param y_max y_max
#' @param always_apply always_apply
#' @param p p
#'
#' @section Targets:
#' image, mask, bboxes, keypoints
#'
#' @section Image types:
#' uint8, float32
#'
#' @export
icevision_Crop <- function(x_min = 0, y_min = 0, x_max = 1024, y_max = 1024, always_apply = FALSE, p = 1.0) {

  icevision()$tfms$albumentations$Crop(
    x_min = as.integer(x_min),
    y_min = as.integer(y_min),
    x_max = as.integer(x_max),
    y_max = as.integer(y_max),
    always_apply = always_apply,
    p = p
  )

}


#' @title Compose
#'
#' @description Compose transforms and handle all transformations regrading bounding boxes
#'
#'
#' @param transforms transforms
#' @param bbox_params bbox_params
#' @param keypoint_params keypoint_params
#' @param additional_targets additional_targets
#' @param p p
#' @return None
#' @export
icevision_Compose <- function(transforms, bbox_params = NULL, keypoint_params = NULL,
                              additional_targets = NULL, p = 1.0) {

  args <- list(
    transforms = transforms,
    bbox_params = bbox_params,
    keypoint_params = keypoint_params,
    additional_targets = additional_targets,
    p = p
  )

  if(is.null(args$bbox_params))
    args$bbox_params <- NULL

  if(is.null(args$keypoint_params))
    args$keypoint_params <- NULL

  if(is.null(args$additional_targets))
    args$additional_targets <- NULL

  do.call(icevision()$tfms$albumentations$Compose, args)

}

#' @title CropNonEmptyMaskIfExists
#'
#' @description Crop area with mask if mask is non-empty, else make random crop.
#'
#'
#' @param height height
#' @param width width
#' @param ignore_values ignore_values
#' @param ignore_channels ignore_channels
#' @param always_apply always_apply
#' @param p p
#'
#' @section Targets:
#' image, mask, bboxes, keypoints
#'
#' @section Image types:
#' uint8, float32
#' @return None
#' @export
icevision_CropNonEmptyMaskIfExists <- function(height, width, ignore_values = NULL,
                                               ignore_channels = NULL, always_apply = FALSE,
                                               p = 1.0) {

  args <- list(
    height = as.integer(height),
    width = as.integer(width),
    ignore_values = ignore_values,
    ignore_channels = ignore_channels,
    always_apply = always_apply,
    p = p
  )

  if(is.null(args$ignore_values))
    args$ignore_values <- NULL

  if(is.null(args$ignore_channels))
    args$ignore_channels <- NULL

  do.call(icevision()$tfms$albumentations$CropNonEmptyMaskIfExists, args)

}


#' @title Cutout
#'
#' @description CoarseDropout of the square regions in the image.
#'
#'
#' @param num_holes num_holes
#' @param max_h_size max_h_size
#' @param max_w_size max_w_size
#' @param fill_value fill_value
#' @param always_apply always_apply
#' @param p p
#'
#' @section Targets:
#' image
#'
#' @section Image types:
#' uint8, float32
#'
#' @section Reference:
#' | https://arxiv.org/abs/1708.04552 | https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py | https://github.com/aleju/imgaug/blob/master/imgaug/augmenters/arithmetic.py
#' @return None
#' @export
icevision_Cutout <- function(num_holes = 8, max_h_size = 8,
                             max_w_size = 8, fill_value = 0,
                             always_apply = FALSE, p = 0.5) {

  args <- list(
    num_holes = as.integer(num_holes),
    max_h_size = as.integer(max_h_size),
    max_w_size = as.integer(max_w_size),
    fill_value = as.integer(fill_value),
    always_apply = always_apply,
    p = p
  )


  do.call(icevision()$tfms$albumentations$Cutout, args)

}


#' @title Downscale
#'
#' @description Decreases image quality by downscaling and upscaling back.
#'
#'
#' @param scale_min scale_min
#' @param scale_max scale_max
#' @param interpolation cv2 interpolation method. cv2.INTER_NEAREST by default
#' @param always_apply always_apply
#' @param p p
#'
#' @section Targets:
#' image
#'
#' @section Image types:
#' uint8, float32
#' @return None
#' @export
icevision_Downscale <- function(scale_min = 0.25, scale_max = 0.25, interpolation = 0, always_apply = FALSE, p = 0.5) {

  args <- list(
    scale_min = scale_min,
    scale_max = scale_max,
    interpolation = as.integer(interpolation),
    always_apply = always_apply,
    p = p
  )

  do.call(icevision()$tfms$albumentations$Downscale, args)

}


#' @title DualIAATransform
#'
#' @description Transform for segmentation task.
#'
#'
#' @param always_apply always_apply
#' @param p p
#' @return None
#' @export
icevision_DualIAATransform <- function(always_apply = FALSE, p = 0.5) {

  icevision()$tfms$albumentations$DualIAATransform(
    always_apply = always_apply,
    p = p
  )

}


#' @title ElasticTransform
#'
#' @description Elastic deformation of images as described in [Simard2003]_ (with modifications).
#'
#' @details Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5 .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for Convolutional Neural Networks applied to Visual Document Analysis", in Proc. of the International Conference on Document Analysis and Recognition, 2003.
#'
#' @param alpha alpha
#' @param sigma sigma
#' @param alpha_affine alpha_affine
#' @param interpolation interpolation
#' @param border_mode border_mode
#' @param value value
#' @param mask_value mask_value
#' @param always_apply always_apply
#' @param approximate approximate
#' @param p p
#'
#' @section Targets:
#' image, mask
#'
#' @section Image types:
#' uint8, float32
#' @return None
#' @export
icevision_ElasticTransform <- function(alpha = 1, sigma = 50, alpha_affine = 50,
                                       interpolation = 1, border_mode = 4,
                                       value = NULL, mask_value = NULL, always_apply = FALSE,
                                       approximate = FALSE, p = 0.5) {

  args <- list(
    alpha = as.integer(alpha),
    sigma = as.integer(sigma),
    alpha_affine = as.integer(alpha_affine),
    interpolation = as.integer(interpolation),
    border_mode = as.integer(border_mode),
    value = value,
    mask_value = mask_value,
    always_apply = always_apply,
    approximate = approximate,
    p = p
  )

  if(is.null(args$value))
    args$value <- NULL
  else
    args$value <- as.integer(args$value)

  if(is.null(args$mask_value))
    args$mask_value <- NULL
  else
    args$mask_value <- as.integer(args$mask_value)

  do.call(icevision()$tfms$albumentations$ElasticTransform, args)

}


#' @title Equalize
#'
#' @description Equalize the image histogram.
#'
#'
#' @param mode mode
#' @param by_channels by_channels
#' @param mask mask
#' @param ... additional arguments
#'
#' @section Targets:
#' image
#'
#' @section Image types:
#' uint8
#' @return None
#' @export
icevision_Equalize <- function(mode = "cv", by_channels = TRUE, mask = NULL,
                               ...) {

  args <- list(
    mode = mode,
    by_channels = by_channels,
    mask = mask,
    ...
  )

  if(is.null(args$mask))
    args$mask <- NULL

  do.call(icevision()$tfms$albumentations$Equalize, args)

}

#' @title FancyPCA
#'
#' @description Augment RGB image using FancyPCA from Krizhevsky's paper
#'
#' @details "ImageNet Classification with Deep Convolutional Neural Networks"
#'
#' @param alpha alpha
#' @param always_apply always_apply
#' @param p p
#'
#' @section Targets:
#' image
#'
#' @section Image types:
#' 3-channel uint8 images only
#'
#' @section Credit:
#' http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf https://deshanadesai.github.io/notes/Fancy-PCA-with-Scikit-Image https://pixelatedbrian.github.io/2018-04-29-fancy_pca/
#' @return None
#' @export
icevision_FancyPCA <- function(alpha = 0.1, always_apply = FALSE, p = 0.5) {

  args <- list(
    alpha = alpha,
    always_apply = always_apply,
    p = p
  )

  do.call(icevision()$tfms$albumentations$FancyPCA, args)

}


#' @title FDA
#'
#' @description Fourier Domain Adaptation from https://github.com/YanchaoYang/FDA
#'
#' @details Simple "style transfer".
#'
#' @param reference_images reference_images
#' @param beta_limit beta_limit
#' @param read_fn read_fn
#' @param always_apply always_apply
#' @param p p
#'
#' @section Fourier Domain Adaptation from https://github.com/YanchaoYang/FDA:
#' Simple "style transfer".
#'
#' @section Targets:
#' image
#'
#' @section Image types:
#' uint8, float32
#'
#' @section Reference:
#' https://github.com/YanchaoYang/FDA https://openaccess.thecvf.com/content_CVPR_2020/papers/Yang_FDA_Fourier_Domain_Adaptation_for_Semantic_Segmentation_CVPR_2020_paper.pdf
#'
#' @section Example:
#' >>> import numpy as np
#' >>> import albumentations as A
#' >>> image = np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)
#' >>> target_image = np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)
#' >>> aug = A.Compose([A.FDA([target_image], p=1, read_fn=lambda x: x)])
#' >>> result = aug(image=image)
#' @return None
#' @export
icevision_FDA <- function(reference_images, beta_limit = 0.1, read_fn = icevision_read_rgb_image(),
                          always_apply = FALSE, p = 0.5) {

  args <- list(
    reference_images = reference_images,
    beta_limit = beta_limit,
    read_fn = read_fn,
    always_apply = always_apply,
    p = p
  )


  do.call(icevision()$tfms$albumentations$FDA, args)

}


#' @title Read_rgb_image
#'
#'
#' @param path path
#' @return None
#' @export
icevision_read_rgb_image <- function(path) {

  if(missing(path)) {
    icevision()$tfms$albumentations$read_rgb_image
  } else {
    icevision()$tfms$albumentations$read_rgb_image(
      path = path
    )
  }

}

#' @title Read_bgr_image
#'
#'
#' @param path path
#' @return None
#' @export
icevision_read_bgr_image <- function(path) {

  if(missing(path)) {
    icevision()$tfms$albumentations$read_bgr_image
  } else {
    icevision()$tfms$albumentations$read_bgr_image(
      path = path
    )
  }

}

#' @title Flip
#'
#' @description Flip the input either horizontally, vertically or both horizontally and vertically.
#'
#'
#' @param always_apply always_apply
#' @param p p
#'
#' @section Targets:
#' image, mask, bboxes, keypoints
#'
#' @section Image types:
#' uint8, float32
#' @return None
#' @export
icevision_Flip <- function(always_apply = FALSE, p = 0.5) {

  icevision()$tfms$albumentations$Flip(
    always_apply = always_apply,
    p = p
  )

}


#' @title FromFloat
#'
#' @description Take an input array where all values should lie in the range [0, 1.0], multiply them by `max_value` and then
#'
#' @details cast the resulted value to a type specified by `dtype`. If `max_value` is NULL the transform will try to infer
#' the maximum value for the data type from the `dtype` argument. This is the inverse transform for :class:`~albumentations.augmentations.transforms.ToFloat`.
#'
#' @param dtype dtype
#' @param max_value max_value
#' @param always_apply always_apply
#' @param p p
#'
#' @section Targets:
#' image
#'
#' @section Image types:
#' float32
#' @return None
#' @export
icevision_FromFloat <- function(dtype = "uint16", max_value = NULL, always_apply = FALSE, p = 1.0) {

  args <- list(
    dtype = dtype,
    max_value = max_value,
    always_apply = always_apply,
    p = p
  )

  if(is.null(args$max_value))
    args$max_value <- NULL
  else
    args$max_value <- as.integer(args$max_value)


  do.call(icevision()$tfms$albumentations$FromFloat, args)

}


#' @title GaussianBlur
#'
#' @description Blur the input image using a Gaussian filter with a random kernel size.
#'
#'
#' @param blur_limit blur_limit
#' @param sigma_limit sigma_limit
#' @param always_apply always_apply
#' @param p p
#'
#' @section Targets:
#' image
#'
#' @section Image types:
#' uint8, float32
#' @return None
#' @export
icevision_GaussianBlur <- function(blur_limit = list(3, 7), sigma_limit = 0, always_apply = FALSE, p = 0.5) {

  args <- list(
    blur_limit = as.list(as.integer(unlist(blur_limit))),
    sigma_limit = as.integer(sigma_limit),
    always_apply = always_apply,
    p = p
  )

  do.call(icevision()$tfms$albumentations$GaussianBlur, args)

}


#' @title GaussNoise
#'
#' @description Apply gaussian noise to the input image.
#'
#'
#' @param var_limit var_limit
#' @param mean mean
#' @param always_apply always_apply
#' @param p p
#'
#' @section Targets:
#' image
#'
#' @section Image types:
#' uint8, float32
#' @return None
#' @export
icevision_GaussNoise <- function(var_limit = list(10.0, 50.0), mean = 0, always_apply = FALSE, p = 0.5) {

  icevision()$tfms$albumentations$GaussNoise(
    var_limit = var_limit,
    mean = as.integer(mean),
    always_apply = always_apply,
    p = p
  )

}

#' @title GlassBlur
#'
#' @description Apply glass noise to the input image.
#'
#'
#' @param sigma sigma
#' @param max_delta max_delta
#' @param iterations iterations
#' @param always_apply always_apply
#' @param mode mode
#' @param p p
#'
#' @section Targets:
#' image
#'
#' @section Image types:
#' uint8, float32
#'
#' @section Reference:
#' | https://arxiv.org/abs/1903.12261 | https://github.com/hendrycks/robustness/blob/master/ImageNet-C/create_c/make_imagenet_c.py
#' @return None
#' @export
icevision_GlassBlur <- function(sigma = 0.7, max_delta = 4, iterations = 2,
                                always_apply = FALSE, mode = "fast", p = 0.5) {

  icevision()$tfms$albumentations$GlassBlur(
    sigma = sigma,
    max_delta = as.integer(max_delta),
    iterations = as.integer(iterations),
    always_apply = always_apply,
    mode = mode,
    p = p
  )

}

#' @title GridDistortion
#'
#' @description Args:
#'
#' @details num_steps (int): count of grid cells on each side. distort_limit (float, (float, float)): If distort_limit is a single float, the range will be (-distort_limit, distort_limit). Default: (-0.03, 0.03). interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of: cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4. Default: cv2.INTER_LINEAR. border_mode (OpenCV flag): flag that is used to specify the pixel extrapolation method. Should be one of: cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT, cv2.BORDER_WRAP, cv2.BORDER_REFLECT_101. Default: cv2.BORDER_REFLECT_101 value (int, float, list of ints, list of float): padding value if border_mode is cv2.BORDER_CONSTANT. mask_value (int, float, list of ints, list of float): padding value if border_mode is cv2.BORDER_CONSTANT applied for masks. Targets: image, mask Image types: uint8, float32
#'
#' @param num_steps num_steps
#' @param distort_limit distort_limit
#' @param interpolation interpolation
#' @param border_mode border_mode
#' @param value value
#' @param mask_value mask_value
#' @param always_apply always_apply
#' @param p p
#'
#' @section Targets:
#' image, mask
#'
#' @section Image types:
#' uint8, float32
#' @return None
#' @export
icevision_GridDistortion <- function(num_steps = 5, distort_limit = 0.3,
                                     interpolation = 1, border_mode = 4,
                                     value = NULL, mask_value = NULL,
                                     always_apply = FALSE, p = 0.5) {

  args <- list(
    num_steps = as.integer(num_steps),
    distort_limit = distort_limit,
    interpolation = as.integer(interpolation),
    border_mode = as.integer(border_mode),
    value = value,
    mask_value = mask_value,
    always_apply = always_apply,
    p = p
  )

  if(is.null(args$value))
    args$value <- NULL
  else
    args$value <- as.integer(args$value)

  if(is.null(args$mask_value))
    args$mask_value <- NULL
  else
    args$mask_value <- as.integer(args$mask_value)

  do.call(icevision()$tfms$albumentations$GridDistortion, args)

}


#' @title GridDropout
#'
#' @description GridDropout, drops out rectangular regions of an image and the corresponding mask in a grid fashion.
#'
#'
#' @param ratio ratio
#' @param unit_size_min unit_size_min
#' @param unit_size_max unit_size_max
#' @param holes_number_x holes_number_x
#' @param holes_number_y holes_number_y
#' @param shift_x shift_x
#' @param shift_y shift_y
#' @param random_offset random_offset
#' @param fill_value fill_value
#' @param mask_fill_value mask_fill_value
#' @param always_apply always_apply
#' @param p p
#'
#' @section Targets:
#' image, mask
#'
#' @section Image types:
#' uint8, float32
#'
#' @section References:
#' https://arxiv.org/abs/2001.04086
#' @return None
#' @export
icevision_GridDropout <- function(ratio = 0.5, unit_size_min = NULL,
                                  unit_size_max = NULL, holes_number_x = NULL,
                                  holes_number_y = NULL, shift_x = 0, shift_y = 0,
                                  random_offset = FALSE, fill_value = 0,
                                  mask_fill_value = NULL, always_apply = FALSE, p = 0.5) {

  args <- list(
    ratio = ratio,
    unit_size_min = unit_size_min,
    unit_size_max = unit_size_max,
    holes_number_x = holes_number_x,
    holes_number_y = holes_number_y,
    shift_x = as.integer(shift_x),
    shift_y = as.integer(shift_y),
    random_offset = random_offset,
    fill_value = as.integer(fill_value),
    mask_fill_value = mask_fill_value,
    always_apply = always_apply,
    p = p
  )

  if(is.null(args$unit_size_min))
    args$unit_size_min <- NULL
  else
    args$unit_size_min <- as.integer(args$unit_size_min)

  if(is.null(args$unit_size_max))
    args$unit_size_max <- NULL
  else
    args$unit_size_max <- as.integer(args$unit_size_max)

  if(is.null(args$holes_number_x))
    args$holes_number_x <- NULL
  else
    args$holes_number_x <- as.integer(args$holes_number_x)

  if(is.null(args$holes_number_y))
    args$holes_number_y <- NULL
  else
    args$holes_number_y <- as.integer(args$holes_number_y)

  if(is.null(args$mask_fill_value))
    args$mask_fill_value <- NULL
  else
    args$mask_fill_value <- as.integer(args$mask_fill_value)

  do.call(icevision()$tfms$albumentations$GridDropout, args)

}


#' @title HistogramMatching
#'
#' @description Apply histogram matching. It manipulates the pixels of an input image so that its histogram matches
#'
#' @details the histogram of the reference image. If the images have multiple channels, the matching is done independently
#' for each channel, as long as the number of channels is equal in the input image and the reference. Histogram matching can be used as a lightweight normalisation for image processing,
#' such as feature matching, especially in circumstances where the images have been taken from different
#' sources or in different conditions (i.e. lighting). See: https://scikit-image.org/docs/dev/auto_examples/color_exposure/plot_histogram_matching.html
#'
#' @param reference_images reference_images
#' @param blend_ratio blend_ratio
#' @param read_fn read_fn
#' @param always_apply always_apply
#' @param p p
#'
#' @section See:
#' https://scikit-image.org/docs/dev/auto_examples/color_exposure/plot_histogram_matching.html
#'
#' @section Targets:
#' image
#'
#' @section Image types:
#' uint8, uint16, float32
#' @return None
#' @export
icevision_HistogramMatching <- function(reference_images, blend_ratio = list(0.5, 1.0),
                                        read_fn = icevision_read_rgb_image(), always_apply = FALSE, p = 0.5) {

  args <- list(
    reference_images = reference_images,
    blend_ratio = blend_ratio,
    read_fn = read_fn,
    always_apply = always_apply,
    p = p
  )

  do.call(icevision()$tfms$albumentations$HistogramMatching, args)

}



#' @title IAAAdditiveGaussianNoise
#'
#' @description Add gaussian noise to the input image.
#'
#'
#' @param loc loc
#' @param scale scale
#' @param per_channel per_channel
#' @param always_apply always_apply
#' @param p p
#'
#' @section Targets:
#' image
#' @return None
#' @export
icevision_IAAAdditiveGaussianNoise <- function(loc = 0,
                                               scale = list(2.5500000000000003, 12.75),
                                               per_channel = FALSE, always_apply = FALSE, p = 0.5) {

  args <- list(
    loc = as.integer(loc),
    scale = scale,
    per_channel = per_channel,
    always_apply = always_apply,
    p = p
  )

  do.call(icevision()$tfms$albumentations$IAAAdditiveGaussianNoise, args)

}

#' @title IAAAffine
#'
#' @description Place a regular grid of points on the input and randomly move the neighbourhood of these point around
#'
#' @details via affine transformations. Note: This class introduce interpolation artifacts to mask if it has values other than {0;1}
#'
#' @param scale scale
#' @param translate_percent translate_percent
#' @param translate_px translate_px
#' @param rotate rotate
#' @param shear shear
#' @param order order
#' @param cval cval
#' @param mode mode
#' @param always_apply always_apply
#' @param p p
#' @return None
#' @section Targets:
#' image, mask
#' @return None
#' @export
icevision_IAAAffine <- function(scale = 1.0, translate_percent = NULL,
                                translate_px = NULL, rotate = 0.0,
                                shear = 0.0, order = 1, cval = 0,
                                mode = "reflect", always_apply = FALSE, p = 0.5) {

  args <- list(
    scale = scale,
    translate_percent = translate_percent,
    translate_px = translate_px,
    rotate = rotate,
    shear = shear,
    order = as.integer(order),
    cval = as.integer(cval),
    mode = mode,
    always_apply = always_apply,
    p = p
  )


  if(is.null(args$translate_px))
    args$translate_px <- NULL

  if(is.null(args$translate_percent))
    args$translate_percent <- NULL


  do.call(icevision()$tfms$albumentations$IAAAffine, args)

}


#' @title IAACropAndPad
#'
#' @description Transform for segmentation task.
#'
#'
#' @param px px
#' @param percent percent
#' @param pad_mode pad_mode
#' @param pad_cval pad_cval
#' @param keep_size keep_size
#' @param always_apply always_apply
#' @param p p
#'
#' @export
icevision_IAACropAndPad <- function(px = NULL, percent = NULL,
                                    pad_mode = "constant", pad_cval = 0,
                                    keep_size = TRUE, always_apply = FALSE, p = 1) {

  args <- list(
    px = px,
    percent = percent,
    pad_mode = pad_mode,
    pad_cval = as.integer(pad_cval),
    keep_size = keep_size,
    always_apply = always_apply,
    p = p
  )


  if(is.null(args$px))
    args$px <- NULL


  if(is.null(args$percent))
    args$percent <- NULL


  do.call(icevision()$tfms$albumentations$IAACropAndPad, args)

}

#' @title IAAEmboss
#'
#' @description Emboss the input image and overlays the result with the original image.
#'
#'
#' @param alpha alpha
#' @param strength strength
#' @param always_apply always_apply
#' @param p p
#'
#' @section Targets:
#' image
#' @return None
#' @export
icevision_IAAEmboss <- function(alpha = list(0.2, 0.5), strength = list(0.2, 0.7),
                                always_apply = FALSE, p = 0.5) {

  icevision()$tfms$albumentations$IAAEmboss(
    alpha = alpha,
    strength = strength,
    always_apply = always_apply,
    p = p
  )

}


#' @title IAAFliplr
#'
#' @description Transform for segmentation task.
#'
#'
#' @param always_apply always_apply
#' @param p p
#' @return None
#' @export
icevision_IAAFliplr <- function(always_apply = FALSE, p = 0.5) {

  icevision()$tfms$albumentations$IAAFliplr(
    always_apply = always_apply,
    p = p
  )

}

#' @title IAAFlipud
#'
#' @description Transform for segmentation task.
#'
#'
#' @param always_apply always_apply
#' @param p p
#' @return None
#' @export
icevision_IAAFlipud <- function(always_apply = FALSE, p = 0.5) {

  icevision()$tfms$albumentations$IAAFlipud(
    always_apply = always_apply,
    p = p
  )

}

#' @title IAAPerspective
#'
#' @description Perform a random four point perspective transform of the input.
#'
#' @details Note: This class introduce interpolation artifacts to mask if it has values other than {0;1}
#'
#' @param scale scale
#' @param keep_size keep_size
#' @param always_apply always_apply
#' @param p p
#'
#' @section Targets:
#' image, mask
#' @return None
#' @export
icevision_IAAPerspective <- function(scale = list(0.05, 0.1), keep_size = TRUE,
                                     always_apply = FALSE, p = 0.5) {

  icevision()$tfms$albumentations$IAAPerspective(
    scale = scale,
    keep_size = keep_size,
    always_apply = always_apply,
    p = p
  )

}

#' @title IAAPiecewiseAffine
#'
#' @description Place a regular grid of points on the input and randomly move the neighbourhood of these point around
#'
#' @details via affine transformations. Note: This class introduce interpolation artifacts to mask if it has values other than {0;1}
#'
#' @param scale scale
#' @param nb_rows nb_rows
#' @param nb_cols nb_cols
#' @param order order
#' @param cval cval
#' @param mode mode
#' @param always_apply always_apply
#' @param p p
#'
#' @section Targets:
#' image, mask
#' @return None
#' @export
icevision_IAAPiecewiseAffine <- function(scale = list(0.03, 0.05),
                                         nb_rows = 4, nb_cols = 4, order = 1, cval = 0,
                                         mode = "constant", always_apply = FALSE, p = 0.5) {

  args <- list(
    scale = scale,
    nb_rows = as.integer(nb_rows),
    nb_cols = as.integer(nb_cols),
    order = as.integer(order),
    cval = as.integer(cval),
    mode = mode,
    always_apply = always_apply,
    p = p
  )


  do.call(icevision()$tfms$albumentations$IAAPiecewiseAffine, args)

}

#' @title IAASharpen
#'
#' @description Sharpen the input image and overlays the result with the original image.
#'
#'
#' @param alpha alpha
#' @param lightness lightness
#' @param always_apply always_apply
#' @param p p
#'
#' @section Targets:
#' image
#' @return None
#' @export
icevision_IAASharpen <- function(alpha = list(0.2, 0.5),
                                 lightness = list(0.5, 1.0), always_apply = FALSE, p = 0.5) {

  icevision()$tfms$albumentations$IAASharpen(
    alpha = alpha,
    lightness = lightness,
    always_apply = always_apply,
    p = p
  )

}

#' @title IAASuperpixels
#'
#' @description Completely or partially transform the input image to its superpixel representation. Uses skimage's version
#'
#' @details of the SLIC algorithm. May be slow.
#'
#' @param p_replace p_replace
#' @param n_segments n_segments
#' @param always_apply always_apply
#' @param p p
#'
#' @section Targets:
#' image
#' @return None
#' @export
icevision_IAASuperpixels <- function(p_replace = 0.1, n_segments = 100,
                                     always_apply = FALSE, p = 0.5) {

  icevision()$tfms$albumentations$IAASuperpixels(
    p_replace = p_replace,
    n_segments = as.integer(n_segments),
    always_apply = always_apply,
    p = p
  )

}


#' @title ImageCompression
#'
#' @description Decrease Jpeg, WebP compression of an image.
#'
#'
#' @param quality_lower quality_lower
#' @param quality_upper quality_upper
#' @param compression_type compression_type
#' @param always_apply always_apply
#' @param p p
#'
#' @section Targets:
#' image
#'
#' @section Image types:
#' uint8, float32
#' @return None
#' @export
icevision_ImageCompression <- function(quality_lower = 99, quality_upper = 100,
                                       compression_type = 0,
                                       always_apply = FALSE, p = 0.5) {

  args <- list(
    quality_lower = as.integer(quality_lower),
    quality_upper = as.integer(quality_upper),
    compression_type = as.integer(compression_type),
    always_apply = always_apply,
    p = p
  )



  do.call(icevision()$tfms$albumentations$ImageCompression, args)

}

#' @title ImageOnlyIAATransform
#'
#' @description Transform applied to image only.
#'
#'
#' @param always_apply always_apply
#' @param p p
#' @return None
#' @export
icevision_ImageOnlyIAATransform <- function(always_apply = FALSE, p = 0.5) {

  icevision()$tfms$albumentations$ImageOnlyIAATransform(
    always_apply = always_apply,
    p = p
  )

}

#' @title ImageOnlyTransform
#'
#' @description Transform applied to image only.
#'
#'
#' @param always_apply always_apply
#' @param p p
#' @return None
#' @export
icevision_ImageOnlyTransform <- function(always_apply = FALSE, p = 0.5) {

  icevision()$tfms$albumentations$ImageOnlyTransform(
    always_apply = always_apply,
    p = p
  )

}


#' @title InvertImg
#'
#' @description Invert the input image by subtracting pixel values from 255.
#'
#'
#' @param always_apply always_apply
#' @param p p
#'
#' @section Targets:
#' image
#'
#' @section Image types:
#' uint8
#' @return None
#' @export
icevision_InvertImg <- function(always_apply = FALSE, p = 0.5) {

  icevision()$tfms$albumentations$InvertImg(
    always_apply = always_apply,
    p = p
  )

}


#' @title ISONoise
#'
#' @description Apply camera sensor noise.
#'
#'
#' @param color_shift color_shift
#' @param intensity intensity
#' @param always_apply always_apply
#' @param p p
#'
#' @section Targets:
#' image
#'
#' @section Image types:
#' uint8
#' @return None
#' @export
icevision_ISONoise <- function(color_shift = list(0.01, 0.05), intensity = list(0.1, 0.5),
                               always_apply = FALSE, p = 0.5) {

  icevision()$tfms$albumentations$ISONoise(
    color_shift = color_shift,
    intensity = intensity,
    always_apply = always_apply,
    p = p
  )

}


#' @title JpegCompression
#'
#' @description Decrease Jpeg compression of an image.
#'
#'
#' @param quality_lower quality_lower
#' @param quality_upper quality_upper
#' @param always_apply always_apply
#' @param p p
#'
#' @section Targets:
#' image
#'
#' @section Image types:
#' uint8, float32
#' @return None
#' @export
icevision_JpegCompression <- function(quality_lower = 99,
                                      quality_upper = 100,
                                      always_apply = FALSE, p = 0.5) {

  icevision()$tfms$albumentations$JpegCompression(
    quality_lower = as.integer(quality_lower),
    quality_upper = as.integer(quality_upper),
    always_apply = always_apply,
    p = p
  )

}

#' @title LongestMaxSize
#'
#' @description Rescale an image so that maximum side is equal to max_size, keeping the aspect ratio of the initial image.
#'
#'
#' @param max_size max_size
#' @param interpolation interpolation
#' @param always_apply always_apply
#' @param p p
#'
#' @section Targets:
#' image, mask, bboxes, keypoints
#'
#' @section Image types:
#' uint8, float32
#' @return None
#' @export
icevision_LongestMaxSize <- function(max_size = 1024, interpolation = 1, always_apply = FALSE, p = 1) {

  icevision()$tfms$albumentations$LongestMaxSize(
    max_size = as.integer(max_size),
    interpolation = as.integer(interpolation),
    always_apply = always_apply,
    p = p
  )

}


#' @title MaskDropout
#'
#' @description Image & mask augmentation that zero out mask and image regions corresponding
#'
#' @details to randomly chosen object instance from mask. Mask must be single-channel image, zero values treated as background.
#' Image can be any number of channels. Inspired by https://www.kaggle.com/c/severstal-steel-defect-detection/discussion/114254
#'
#' @param max_objects max_objects
#' @param image_fill_value image_fill_value
#' @param mask_fill_value mask_fill_value
#' @param always_apply always_apply
#' @param p p
#' @return None
#' @export
icevision_MaskDropout <- function(max_objects = 1, image_fill_value = 0,
                                  mask_fill_value = 0, always_apply = FALSE, p = 0.5) {

  icevision()$tfms$albumentations$MaskDropout(
    max_objects = as.integer(max_objects),
    image_fill_value = as.integer(image_fill_value),
    mask_fill_value = as.integer(mask_fill_value),
    always_apply = always_apply,
    p = p
  )

}


#' @title MedianBlur
#'
#' @description Blur the input image using a median filter with a random aperture linear size.
#'
#'
#' @param blur_limit blur_limit
#' @param always_apply always_apply
#' @param p p
#'
#' @section Targets:
#' image
#'
#' @section Image types:
#' uint8, float32
#' @return None
#' @export
icevision_MedianBlur <- function(blur_limit = 7, always_apply = FALSE, p = 0.5) {

  icevision()$tfms$albumentations$MedianBlur(
    blur_limit = as.integer(blur_limit),
    always_apply = always_apply,
    p = p
  )

}

#' @title MotionBlur
#'
#' @description Apply motion blur to the input image using a random-sized kernel.
#'
#'
#' @param blur_limit blur_limit
#' @param always_apply always_apply
#' @param p p
#'
#' @section Targets:
#' image
#'
#' @section Image types:
#' uint8, float32
#' @return None
#' @export
icevision_MotionBlur <- function(blur_limit = 7, always_apply = FALSE, p = 0.5) {

  icevision()$tfms$albumentations$MotionBlur(
    blur_limit = as.integer(blur_limit),
    always_apply = always_apply,
    p = p
  )

}


#' @title MultiplicativeNoise
#'
#' @description Multiply image to random number or array of numbers.
#'
#'
#' @param multiplier multiplier
#' @param per_channel per_channel
#' @param elementwise elementwise
#' @param always_apply always_apply
#' @param p p
#'
#' @section Targets:
#' image
#'
#' @section Image types:
#' Any
#' @return None
#' @export
icevision_MultiplicativeNoise <- function(multiplier = list(0.9, 1.1),
                                          per_channel = FALSE, elementwise = FALSE,
                                          always_apply = FALSE, p = 0.5) {

  icevision()$tfms$albumentations$MultiplicativeNoise(
    multiplier = multiplier,
    per_channel = per_channel,
    elementwise = elementwise,
    always_apply = always_apply,
    p = p
  )

}

#' @title Normalize
#'
#' @description Divide pixel values by 255 = 2**8 - 1, subtract mean per channel and divide by std per channel.
#'
#'
#' @param mean mean
#' @param std std
#' @param max_pixel_value max_pixel_value
#' @param always_apply always_apply
#' @param p p
#'
#' @section Targets:
#' image
#'
#' @section Image types:
#' uint8, float32
#' @return None
#' @export
icevision_Normalize <- function(mean = list(0.485, 0.456, 0.406),
                                std = list(0.229, 0.224, 0.225),
                                max_pixel_value = 255.0, always_apply = FALSE, p = 1.0) {

  icevision()$tfms$albumentations$Normalize(
    mean = mean,
    std = std,
    max_pixel_value = max_pixel_value,
    always_apply = always_apply,
    p = p
  )

}

#' @title OpticalDistortion
#'
#'
#' @details distort_limit (float, (float, float)): If distort_limit is a single float, the range will be (-distort_limit, distort_limit). Default: (-0.05, 0.05). shift_limit (float, (float, float))): If shift_limit is a single float, the range will be (-shift_limit, shift_limit). Default: (-0.05, 0.05). interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of: cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4. Default: cv2.INTER_LINEAR. border_mode (OpenCV flag): flag that is used to specify the pixel extrapolation method. Should be one of: cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT, cv2.BORDER_WRAP, cv2.BORDER_REFLECT_101. Default: cv2.BORDER_REFLECT_101 value (int, float, list of ints, list of float): padding value if border_mode is cv2.BORDER_CONSTANT. mask_value (int, float, list of ints, list of float): padding value if border_mode is cv2.BORDER_CONSTANT applied for masks. Targets: image, mask Image types: uint8, float32
#'
#' @param distort_limit distort_limit
#' @param shift_limit shift_limit
#' @param interpolation interpolation
#' @param border_mode border_mode
#' @param value value
#' @param mask_value mask_value
#' @param always_apply always_apply
#' @param p p
#'
#' @section Targets:
#' image, mask
#'
#' @section Image types:
#' uint8, float32
#' @return None
#' @export
icevision_OpticalDistortion <- function(distort_limit = 0.05, shift_limit = 0.05,
                                        interpolation = 1, border_mode = 4,
                                        value = NULL, mask_value = NULL, always_apply = FALSE, p = 0.5) {

  args <- list(
    distort_limit = distort_limit,
    shift_limit = shift_limit,
    interpolation = as.integer(interpolation),
    border_mode = as.integer(border_mode),
    value = value,
    mask_value = mask_value,
    always_apply = always_apply,
    p = p
  )

  if(is.null(args$value))
    args$value <- NULL
  else
    args$value <- as.integer(args$value)

  if(is.null(args$mask_value))
    args$mask_value <- NULL
  else
    args$mask_value <- as.integer(args$mask_value)


  do.call(icevision()$tfms$albumentations$OpticalDistortion, args)

}


#' @title Posterize
#'
#' @description Reduce the number of bits for each color channel.
#'
#'
#' @param num_bits num_bits
#' @param always_apply always_apply
#' @param p p
#'
#' @section Targets:
#' image
#'
#' @section Image types:
#' uint8
#' @return None
#' @export
icevision_Posterize <- function(num_bits = 4, always_apply = FALSE, p = 0.5) {

  icevision()$tfms$albumentations$Posterize(
    num_bits = as.integer(num_bits),
    always_apply = always_apply,
    p = p
  )

}


#' @title RandomContrast
#'
#' @description Randomly change contrast of the input image.
#'
#'
#' @param limit limit
#' @param always_apply always_apply
#' @param p p
#'
#' @section Targets:
#' image
#'
#' @section Image types:
#' uint8, float32
#' @return None
#' @export
icevision_RandomContrast <- function(limit = 0.2, always_apply = FALSE, p = 0.5) {

  icevision()$tfms$albumentations$RandomContrast(
    limit = limit,
    always_apply = always_apply,
    p = p
  )

}

#' @title RandomCrop
#'
#' @description Crop a random part of the input.
#'
#'
#' @param height height
#' @param width width
#' @param always_apply always_apply
#' @param p p
#'
#' @section Targets:
#' image, mask, bboxes, keypoints
#'
#' @section Image types:
#' uint8, float32
#' @return None
#' @export
icevision_RandomCrop <- function(height, width, always_apply = FALSE, p = 1.0) {

  icevision()$tfms$albumentations$RandomCrop(
    height = as.integer(height),
    width = as.integer(width),
    always_apply = always_apply,
    p = p
  )

}


#' @title RandomCropNearBBox
#'
#' @description Crop bbox from image with random shift by x,y coordinates
#'
#'
#' @param max_part_shift max_part_shift
#' @param always_apply always_apply
#' @param p p
#'
#' @section Targets:
#' image, mask, bboxes, keypoints
#'
#' @section Image types:
#' uint8, float32
#' @return None
#' @export
icevision_RandomCropNearBBox <- function(max_part_shift = 0.3, always_apply = FALSE, p = 1.0) {

  icevision()$tfms$albumentations$RandomCropNearBBox(
    max_part_shift = max_part_shift,
    always_apply = always_apply,
    p = p
  )

}

#' @title RandomFog
#'
#' @description Simulates fog for the image
#'
#' @details From https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library
#'
#' @param fog_coef_lower fog_coef_lower
#' @param fog_coef_upper fog_coef_upper
#' @param alpha_coef alpha_coef
#' @param always_apply always_apply
#' @param p p
#'
#' @section Targets:
#' image
#'
#' @section Image types:
#' uint8, float32
#' @return None
#' @export
icevision_RandomFog <- function(fog_coef_lower = 0.3, fog_coef_upper = 1,
                                alpha_coef = 0.08, always_apply = FALSE, p = 0.5) {

  icevision()$tfms$albumentations$RandomFog(
    fog_coef_lower = fog_coef_lower,
    fog_coef_upper = fog_coef_upper,
    alpha_coef = alpha_coef,
    always_apply = always_apply,
    p = p
  )

}

#' @title RandomGamma
#'
#'
#' @details gamma_limit (float or (float, float)): If gamma_limit is a single float value, the range will be (-gamma_limit, gamma_limit). Default: (80, 120). eps: Deprecated. Targets: image Image types: uint8, float32
#'
#' @param gamma_limit gamma_limit
#' @param eps Deprecated.
#' @param always_apply always_apply
#' @param p p
#'
#' @section Targets:
#' image
#'
#' @section Image types:
#' uint8, float32
#' @return None
#' @export
icevision_RandomGamma <- function(gamma_limit = list(80, 120), eps = NULL, always_apply = FALSE, p = 0.5) {

  icevision()$tfms$albumentations$RandomGamma(
    gamma_limit = as.list(as.integer(unlist(gamma_limit))),
    eps = eps,
    always_apply = always_apply,
    p = p
  )

}

#' @title RandomGridShuffle
#'
#' @description Random shuffle grid's cells on image.
#'
#'
#' @param grid grid
#' @param always_apply always_apply
#' @param p p
#'
#' @section Targets:
#' image, mask
#'
#' @section Image types:
#' uint8, float32
#' @return None
#' @export
icevision_RandomGridShuffle <- function(grid = list(3, 3), always_apply = FALSE, p = 0.5) {

  icevision()$tfms$albumentations$RandomGridShuffle(
    grid = as.list(as.integer(unlist(grid))),
    always_apply = always_apply,
    p = p
  )

}


#' @title RandomRain
#'
#' @description Adds rain effects.
#'
#' @details From https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library
#'
#' @param slant_lower should be in range [-20, 20].
#' @param slant_upper should be in range [-20, 20].
#' @param drop_length should be in range [0, 100].
#' @param drop_width should be in range [1, 5]. drop_color (list of (r, g, b)): rain lines color. blur_value (int): rainy view are blurry brightness_coefficient (float): rainy days are usually shady. Should be in range [0, 1].
#' @param drop_color drop_color
#' @param blur_value blur_value
#' @param brightness_coefficient brightness_coefficient
#' @param rain_type One of [NULL, "drizzle", "heavy", "torrestial"]
#' @param always_apply always_apply
#' @param p p
#'
#' @section Targets:
#' image
#'
#' @section Image types:
#' uint8, float32
#' @return None
#' @export
icevision_RandomRain <- function(slant_lower = -10, slant_upper = 10, drop_length = 20,
                                 drop_width = 1, drop_color = list(200, 200, 200),
                                 blur_value = 7, brightness_coefficient = 0.7,
                                 rain_type = NULL, always_apply = FALSE, p = 0.5) {

  args <- list(
    slant_lower = as.integer(slant_lower),
    slant_upper = as.integer(slant_upper),
    drop_length = as.integer(drop_length),
    drop_width = as.integer(drop_width),
    drop_color = as.list(as.integer(unlist(drop_color))),
    blur_value = as.integer(blur_value),
    brightness_coefficient = brightness_coefficient,
    rain_type = rain_type,
    always_apply = always_apply,
    p = p
  )

  if(is.null(args$rain_type))
    args$rain_type <- NULL




  do.call(icevision()$tfms$albumentations$RandomRain, args)

}


#' @title RandomResizedCrop
#'
#' @description Torchvision's variant of crop a random part of the input and rescale it to some size.
#'
#'
#' @param height height
#' @param width width
#' @param scale scale
#' @param ratio ratio
#' @param interpolation interpolation
#' @param always_apply always_apply
#' @param p p
#'
#' @section Targets:
#' image, mask, bboxes, keypoints
#'
#' @section Image types:
#' uint8, float32
#' @return None
#' @export
icevision_RandomResizedCrop <- function(height, width, scale = list(0.08, 1.0),
                                        ratio = list(0.75, 1.3333333333333333),
                                        interpolation = 1, always_apply = FALSE, p = 1.0) {

  args <- list(
    height = as.integer(height),
    width = as.integer(width),
    scale = scale,
    ratio = ratio,
    interpolation = as.integer(interpolation),
    always_apply = always_apply,
    p = p
  )



  do.call(icevision()$tfms$albumentations$RandomResizedCrop, args)

}

#' @title RandomRotate90
#'
#' @description Randomly rotate the input by 90 degrees zero or more times.
#'
#'
#' @param always_apply always_apply
#' @param p p
#'
#' @section Targets:
#' image, mask, bboxes, keypoints
#'
#' @section Image types:
#' uint8, float32
#' @return None
#' @export
icevision_RandomRotate90 <- function(always_apply = FALSE, p = 0.5) {

  icevision()$tfms$albumentations$RandomRotate90(
    always_apply = always_apply,
    p = p
  )

}


#' @title RandomScale
#'
#' @description Randomly resize the input. Output image size is different from the input image size.
#'
#'
#' @param scale_limit scale_limit
#' @param interpolation interpolation
#' @param always_apply always_apply
#' @param p p
#'
#' @section Targets:
#' image, mask, bboxes, keypoints
#'
#' @section Image types:
#' uint8, float32
#' @return None
#' @export
icevision_RandomScale <- function(scale_limit = 0.1, interpolation = 1L, always_apply = FALSE, p = 0.5) {

  icevision()$tfms$albumentations$RandomScale(
    scale_limit = scale_limit,
    interpolation = interpolation,
    always_apply = always_apply,
    p = p
  )

}


#' @title RandomShadow
#'
#' @description Simulates shadows for the image
#'
#' @details From https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library
#'
#' @param shadow_roi shadow_roi
#' @param num_shadows_lower num_shadows_lower
#' @param num_shadows_upper num_shadows_upper
#' @param shadow_dimension shadow_dimension
#' @param always_apply always_apply
#' @param p p
#'
#' @section Targets:
#' image
#'
#' @section Image types:
#' uint8, float32
#' @return None
#' @export
icevision_RandomShadow <- function(shadow_roi = list(0, 0.5, 1, 1), num_shadows_lower = 1,
                                   num_shadows_upper = 2, shadow_dimension = 5, always_apply = FALSE, p = 0.5) {

  icevision()$tfms$albumentations$RandomShadow(
    shadow_roi = shadow_roi,
    num_shadows_lower = as.integer(num_shadows_lower),
    num_shadows_upper = as.integer(num_shadows_upper),
    shadow_dimension = as.integer(shadow_dimension),
    always_apply = always_apply,
    p = p
  )

}

#' @title RandomSizedBBoxSafeCrop
#'
#' @description Crop a random part of the input and rescale it to some size without loss of bboxes.
#'
#'
#' @param height height
#' @param width width
#' @param erosion_rate erosion_rate
#' @param interpolation interpolation
#' @param always_apply always_apply
#' @param p p
#'
#' @section Targets:
#' image, mask, bboxes
#'
#' @section Image types:
#' uint8, float32
#' @return None
#' @export
icevision_RandomSizedBBoxSafeCrop <- function(height, width, erosion_rate = 0.0,
                                              interpolation = 1, always_apply = FALSE, p = 1.0) {

  icevision()$tfms$albumentations$RandomSizedBBoxSafeCrop(
    height = as.integer(height),
    width = as.integer(width),
    erosion_rate = erosion_rate,
    interpolation = as.integer(interpolation),
    always_apply = always_apply,
    p = p
  )

}

#' @title RandomSizedCrop
#'
#' @description Crop a random part of the input and rescale it to some size.
#'
#'
#' @param min_max_height min_max_height
#' @param height height
#' @param width width
#' @param w2h_ratio w2h_ratio
#' @param interpolation interpolation
#' @param always_apply always_apply
#' @param p p
#'
#' @section Targets:
#' image, mask, bboxes, keypoints
#'
#' @section Image types:
#' uint8, float32
#' @return None
#' @export
icevision_RandomSizedCrop <- function(min_max_height, height, width, w2h_ratio = 1.0,
                                      interpolation = 1, always_apply = FALSE, p = 1.0) {

  icevision()$tfms$albumentations$RandomSizedCrop(
    min_max_height = as.integer(min_max_height),
    height = as.integer(height),
    width = as.integer(width),
    w2h_ratio = w2h_ratio,
    interpolation = as.integer(interpolation),
    always_apply = always_apply,
    p = p
  )

}

#' @title RandomSnow
#'
#' @description Bleach out some pixel values simulating snow.
#'
#' @details From https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library
#'
#' @param snow_point_lower snow_point_lower
#' @param snow_point_upper snow_point_upper
#' @param brightness_coeff brightness_coeff
#' @param always_apply always_apply
#' @param p p
#'
#' @section Targets:
#' image
#'
#' @section Image types:
#' uint8, float32
#' @return None
#' @export
icevision_RandomSnow <- function(snow_point_lower = 0.1, snow_point_upper = 0.3,
                                 brightness_coeff = 2.5, always_apply = FALSE, p = 0.5) {

  icevision()$tfms$albumentations$RandomSnow(
    snow_point_lower = snow_point_lower,
    snow_point_upper = snow_point_upper,
    brightness_coeff = brightness_coeff,
    always_apply = always_apply,
    p = p
  )

}

#' @title RandomSunFlare
#'
#' @description Simulates Sun Flare for the image
#'
#' @details From https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library
#'
#' @param flare_roi flare_roi
#' @param angle_lower angle_lower
#' @param angle_upper angle_upper
#' @param num_flare_circles_lower num_flare_circles_lower
#' @param num_flare_circles_upper num_flare_circles_upper
#' @param src_radius src_radius
#' @param src_color src_color
#' @param always_apply always_apply
#' @param p p
#'
#' @section Targets:
#' image
#'
#' @section Image types:
#' uint8, float32
#' @return None
#' @export
icevision_RandomSunFlare <- function(flare_roi = list(0, 0, 1, 0.5),
                                     angle_lower = 0, angle_upper = 1,
                                     num_flare_circles_lower = 6,
                                     num_flare_circles_upper = 10,
                                     src_radius = 400,
                                     src_color = list(255, 255, 255),
                                     always_apply = FALSE, p = 0.5) {

  icevision()$tfms$albumentations$RandomSunFlare(
    flare_roi = flare_roi,
    angle_lower = as.integer(angle_lower),
    angle_upper = as.integer(angle_upper),
    num_flare_circles_lower = as.integer(num_flare_circles_lower),
    num_flare_circles_upper = as.integer(num_flare_circles_upper),
    src_radius = as.integer(src_radius),
    src_color = as.list(as.integer(unlist(src_color))),
    always_apply = always_apply,
    p = p
  )

}


#' @title Resize
#'
#' @description Resize the input to the given height and width.
#'
#'
#' @param height height
#' @param width width
#' @param interpolation interpolation
#' @param always_apply always_apply
#' @param p p
#'
#' @section Targets:
#' image, mask, bboxes, keypoints
#'
#' @section Image types:
#' uint8, float32
#' @return None
#' @export
icevision_Resize <- function(height, width, interpolation = 1, always_apply = FALSE, p = 1) {

  icevision()$tfms$albumentations$Resize(
    height = as.integer(height),
    width = as.integer(width),
    interpolation = as.integer(interpolation),
    always_apply = always_apply,
    p = p
  )

}

#' @title RGBShift
#'
#' @description Randomly shift values for each channel of the input RGB image.
#'
#'
#' @param r_shift_limit r_shift_limit
#' @param g_shift_limit g_shift_limit
#' @param b_shift_limit b_shift_limit
#' @param always_apply always_apply
#' @param p p
#'
#' @section Targets:
#' image
#'
#' @section Image types:
#' uint8, float32
#' @return None
#' @export
icevision_RGBShift <- function(r_shift_limit = 20,
                               g_shift_limit = 20,
                               b_shift_limit = 20,
                               always_apply = FALSE, p = 0.5) {

  icevision()$tfms$albumentations$RGBShift(
    r_shift_limit = as.integer(r_shift_limit),
    g_shift_limit = as.integer(g_shift_limit),
    b_shift_limit = as.integer(b_shift_limit),
    always_apply = always_apply,
    p = p
  )

}

#' @title Rotate
#'
#' @description Rotate the input by an angle selected randomly from the uniform distribution.
#'
#'
#' @param limit limit
#' @param interpolation interpolation
#' @param border_mode border_mode
#' @param value value
#' @param mask_value mask_value
#' @param always_apply always_apply
#' @param p p
#'
#' @section Targets:
#' image, mask, bboxes, keypoints
#'
#' @section Image types:
#' uint8, float32
#' @return None
#' @export
icevision_Rotate <- function(limit = 90, interpolation = 1,
                             border_mode = 4, value = NULL,
                             mask_value = NULL, always_apply = FALSE, p = 0.5) {

  args <- list(
    limit = as.integer(limit),
    interpolation = as.integer(interpolation),
    border_mode = as.integer(border_mode),
    value = value,
    mask_value = mask_value,
    always_apply = always_apply,
    p = p
  )

  if(is.null(args$value))
    args$value <- NULL
  else
    args$value <- as.integer(args$value)

  if(is.null(args$mask_value))
    args$mask_value <- NULL
  else
    args$mask_value <- as.integer(args$mask_value)


  do.call(icevision()$tfms$albumentations$Rotate, args)

}


#' @title ShiftScaleRotate
#'
#' @description Randomly apply affine transforms: translate, scale and rotate the input.
#'
#'
#' @param shift_limit shift_limit
#' @param scale_limit scale_limit
#' @param rotate_limit rotate_limit
#' @param interpolation interpolation
#' @param border_mode border_mode
#' @param value value
#' @param mask_value mask_value
#' @param shift_limit_x shift_limit_x
#' @param shift_limit_y shift_limit_y
#' @param always_apply always_apply
#' @param p p
#'
#' @section Targets:
#' image, mask, keypoints
#'
#' @section Image types:
#' uint8, float32
#' @return None
#' @export
icevision_ShiftScaleRotate <- function(shift_limit = 0.0625, scale_limit = 0.1,
                                       rotate_limit = 45, interpolation = 1,
                                       border_mode = 4, value = NULL, mask_value = NULL,
                                       shift_limit_x = NULL, shift_limit_y = NULL, always_apply = FALSE, p = 0.5) {

  args <- list(
    shift_limit = shift_limit,
    scale_limit = scale_limit,
    rotate_limit = as.integer(rotate_limit),
    interpolation = as.integer(interpolation),
    border_mode = as.integer(border_mode),
    value = value,
    mask_value = mask_value,
    shift_limit_x = shift_limit_x,
    shift_limit_y = shift_limit_y,
    always_apply = always_apply,
    p = p
  )

  if(is.null(args$value))
    args$value <- NULL
  else
    args$value <- as.integer(args$value)

  if(is.null(args$mask_value))
    args$mask_value <- NULL
  else
    args$mask_value <- as.integer(args$mask_value)

  if(is.null(args$shift_limit_x))
    args$shift_limit_x <- NULL
  else
    args$shift_limit_x <- as.integer(args$shift_limit_x)

  if(is.null(args$shift_limit_y))
    args$shift_limit_y <- NULL
  else
    args$shift_limit_y <- as.integer(args$shift_limit_y)

  do.call(icevision()$tfms$albumentations$ShiftScaleRotate, args)

}

#' @title SmallestMaxSize
#'
#' @description Rescale an image so that minimum side is equal to max_size, keeping the aspect ratio of the initial image.
#'
#'
#' @param max_size max_size
#' @param interpolation interpolation
#' @param always_apply always_apply
#' @param p p
#'
#' @section Targets:
#' image, mask, bboxes, keypoints
#'
#' @section Image types:
#' uint8, float32
#' @return None
#' @export
icevision_SmallestMaxSize <- function(max_size = 1024, interpolation = 1, always_apply = FALSE, p = 1) {

  icevision()$tfms$albumentations$SmallestMaxSize(
    max_size = as.integer(max_size),
    interpolation = as.integer(interpolation),
    always_apply = always_apply,
    p = p
  )

}

#' @title Solarize
#'
#' @description Invert all pixel values above a threshold.
#'
#' @param threshold threshold
#' @param always_apply always_apply
#' @param p p
#'
#' @section Targets:
#' image
#'
#' @section Image types:
#' any
#' @return None
#' @export
icevision_Solarize <- function(threshold = 128, always_apply = FALSE, p = 0.5) {

  icevision()$tfms$albumentations$Solarize(
    threshold = as.integer(threshold),
    always_apply = always_apply,
    p = p
  )

}

#' @title ToFloat
#'
#' @description Divide pixel values by `max_value` to get a float32 output array where all values lie in the range [0, 1.0].
#'
#' @details If `max_value` is NULL the transform will try to infer the maximum value by inspecting the data type of the input
#' image. See Also: :class:`~albumentations.augmentations.transforms.FromFloat`
#'
#' @param max_value max_value
#' @param always_apply always_apply
#' @param p p
#'
#' @section See Also:
#' :class:`~albumentations.augmentations.transforms.FromFloat`
#'
#' @section Targets:
#' image
#'
#' @section Image types:
#' any type
#' @return None
#' @export
icevision_ToFloat <- function(max_value = NULL, always_apply = FALSE, p = 1.0) {

  args <- list(
    max_value = max_value,
    always_apply = always_apply,
    p = p
  )

  if(is.null(args$max_value))
    args$max_value <- NULL
  else
    args$max_value <- as.integer(args$max_value)

  do.call(icevision()$tfms$albumentations$ToFloat, args)

}

#' @title ToGray
#'
#' @description Convert the input RGB image to grayscale. If the mean pixel value for the resulting image is greater
#' than 127, invert the resulting grayscale image.
#'
#' @param always_apply always_apply
#' @param p p
#'
#' @section Targets:
#' image
#'
#' @section Image types:
#' uint8, float32
#' @return None
#' @export
icevision_ToGray <- function(always_apply = FALSE, p = 0.5) {

  icevision()$tfms$albumentations$ToGray(
    always_apply = always_apply,
    p = p
  )

}

#' @title ToSepia
#'
#' @description Applies sepia filter to the input RGB image
#'
#'
#' @param always_apply always_apply
#' @param p p
#'
#' @section Targets:
#' image
#'
#' @section Image types:
#' uint8, float32
#' @return None
#' @export
icevision_ToSepia <- function(always_apply = FALSE, p = 0.5) {

  icevision()$tfms$albumentations$ToSepia(
    always_apply = always_apply,
    p = p
  )

}


#' @title Transpose
#'
#' @description Transpose the input by swapping rows and columns.
#'
#'
#' @param always_apply always_apply
#' @param p p
#'
#' @section Targets:
#' image, mask, bboxes, keypoints
#'
#' @section Image types:
#' uint8, float32
#' @return None
#' @export
icevision_Transpose <- function(always_apply = FALSE, p = 0.5) {

  icevision()$tfms$albumentations$Transpose(
    always_apply = always_apply,
    p = p
  )

}

#' @title VerticalFlip
#'
#' @description Flip the input vertically around the x-axis.
#'
#'
#' @param always_apply always_apply
#' @param p p
#'
#' @section Targets:
#' image, mask, bboxes, keypoints
#'
#' @section Image types:
#' uint8, float32
#' @return None
#' @export
icevision_VerticalFlip <- function(always_apply = FALSE, p = 0.5) {

  icevision()$tfms$albumentations$VerticalFlip(
    always_apply = always_apply,
    p = p
  )

}


#' @title Resize_and_pad
#'
#'
#' @param size size
#' @param pad pad
#' @return None
#' @export
icevision_resize_and_pad <- function(size,
                                     pad = partial(icevision_PadIfNeeded, border_mode=0, value=c(124L, 116L, 104L))) {

  icevision()$tfms$albumentations$resize_and_pad(
    size = as.integer(size),
    pad = pad
  )

}



