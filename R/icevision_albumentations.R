

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
icevision_aug_tfms <- function(size, presize = NULL, horizontal_flip = HorizontalFlip(always_apply=FALSE, p=0.5),
                     shift_scale_rotate = ShiftScaleRotate(always_apply=FALSE, p=0.5,
                                                           shift_limit_x=c(-0.0625, 0.0625),
                                                           shift_limit_y=c(-0.0625, 0.0625),
                                                           scale_limit=c(-0.09999999999999998,
                                                                         0.10000000000000009),
                                                           rotate_limit=c(-45, 45), interpolation=1,
                                                           border_mode=4, value=NULL, mask_value=NULL),
                     rgb_shift = RGBShift(always_apply=FALSE, p=0.5, r_shift_limit=c(-20, 20),
                                          g_shift_limit=c(-20, 20), b_shift_limit=c(-20, 20)),
                     lightning = RandomBrightnessContrast(always_apply=FALSE, p=0.5,
                                                          brightness_limit=c(-0.2, 0.2),
                                                          contrast_limit=c(-0.2, 0.2),
                                                          brightness_by_max=TRUE),
                     blur = Blur(always_apply=FALSE, p=0.5, blur_limit=c(1, 3)),
                     crop_fn = partial(RandomSizedBBoxSafeCrop, p=0.5),
                     pad = partial(PadIfNeeded, border_mode=0, value=list(124, 116, 104))) {

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
HorizontalFlip <- function(always_apply = FALSE, p = 0.5) {

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
ShiftScaleRotate <- function(shift_limit = 0.0625, scale_limit = 0.1, rotate_limit = 45,
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
RGBShift <- function(r_shift_limit = 20, g_shift_limit = 20, b_shift_limit = 20,
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
RandomBrightnessContrast <- function(brightness_limit = 0.2, contrast_limit = 0.2,
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
Blur <- function(blur_limit = 7, always_apply = FALSE, p = 0.5) {

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
DualTransform <- function(always_apply = FALSE, p = 0.5) {

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
RandomSizedBBoxSafeCrop <- function(height, width, erosion_rate = 0.0, interpolation = 1,
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
PadIfNeeded <- function(min_height = 1024, min_width = 1024, pad_height_divisor = NULL,
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
HueSaturationValue <- function(hue_shift_limit = 20,
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








