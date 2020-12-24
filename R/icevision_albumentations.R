

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
#' @details
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
#' @details
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
#' @details
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






