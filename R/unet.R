#' @title Unet_config
#'
#' @description Convenience function to easily create a config for `DynamicUnet`
#'
#'
#' @param blur blur is used to avoid checkerboard artifacts at each layer.
#' @param blur_final blur final is specific to the last layer.
#' @param self_attention self_attention determines if we use a self attention layer at the third block before the end.
#' @param y_range If y_range is passed, the last activations go through a sigmoid rescaled to that range.
#' @param last_cross last cros
#' @param bottle bottle
#' @param act_cls activation
#' @param init initializer
#' @param norm_type normalization type
#' @return None
#' @export
unet_config <- function(blur = FALSE, blur_final = TRUE, self_attention = FALSE,
                        y_range = NULL, last_cross = TRUE, bottle = FALSE,
                        act_cls = nn()$ReLU, init = nn()$init$kaiming_normal_, norm_type = NULL) {

  args <- list(
    blur = blur,
    blur_final = blur_final,
    self_attention = self_attention,
    y_range = y_range,
    last_cross = last_cross,
    bottle = bottle,
    act_cls = act_cls,
    init = init,
    norm_type = norm_type
  )

  do.call(vision()$gan$unet_config, args)

}




#' @title Unet_learner
#'
#' @description Build a unet learner from `dls` and `arch`
#'
#' @param dls dataloader
#' @param arch architecture
#' @param ... additional arguments
#' @return None
#' @export
unet_learner <- function(dls, arch, ...) {

  args <- list(
    dls = dls,
    arch = arch,
    ...
  )

  if(!is.null(args[['n_in']])) {
    args[['n_in']] = as.integer(args[['n_in']])
  }
  do.call(vision()$gan$unet_learner, args)

}


#' @title UnetBlock
#'
#' @description A quasi-UNet block, using `PixelShuffle_ICNR upsampling`.
#'
#'
#' @param up_in_c up_in_c parameter
#' @param x_in_c x_in_c parameter
#' @param hook The hook is set to this intermediate layer to store the output needed for this block.
#' @param final_div final div
#' @param blur blur is used to avoid checkerboard artifacts at each layer.
#' @param act_cls activation
#' @param self_attention self_attention determines if we use a self-attention layer
#' @param init initializer
#' @param norm_type normalization type
#' @param ks kernel size
#' @param stride stride
#' @param padding padding mode
#' @param bias bias
#' @param ndim number of dimensions
#' @param bn_1st batch normalization 1st
#' @param transpose transpose
#' @param xtra xtra
#' @param bias_std bias standard deviation
#' @param dilation dilation
#' @param groups groups
#' @param padding_mode The mode of padding
#' @return None
#' @export
UnetBlock <- function(up_in_c, x_in_c, hook, final_div = TRUE,
                      blur = FALSE, act_cls = nn()$ReLU, self_attention = FALSE,
                      init = nn()$init$kaiming_normal_, norm_type = NULL, ks = 3,
                      stride = 1, padding = NULL, bias = NULL, ndim = 2,
                      bn_1st = TRUE, transpose = FALSE, xtra = NULL, bias_std = 0.01,
                      dilation = 1, groups = 1, padding_mode = "zeros") {

  args <- list(
    up_in_c = up_in_c,
    x_in_c = x_in_c,
    hook = hook,
    final_div = final_div,
    blur = blur,
    act_cls = act_cls,
    self_attention = self_attention,
    init = init,
    norm_type = norm_type,
    ks = as.integer(ks),
    stride = as.integer(stride),
    padding = padding,
    bias = bias,
    ndim = as.integer(ndim),
    bn_1st = bn_1st,
    transpose = transpose,
    xtra = xtra,
    bias_std = bias_std,
    dilation = as.integer(dilation),
    groups = as.integer(groups),
    padding_mode = padding_mode
  )

  do.call(vision()$gan$UnetBlock, args)

}


#' @title DynamicUnet
#'
#' @description Create a U-Net from a given architecture.
#'
#'
#' @param encoder encoder
#' @param n_classes number of classes
#' @param img_size image size
#' @param blur blur is used to avoid checkerboard artifacts at each layer.
#' @param blur_final blur final is specific to the last layer.
#' @param self_attention self_attention determines if we use a self attention layer at the third block before the end.
#' @param y_range If y_range is passed, the last activations go through a sigmoid rescaled to that range.
#' @param last_cross last cross
#' @param bottle bottle
#' @param act_cls activation
#' @param init initializer
#' @param norm_type normalization type
#' @return None
#' @export
DynamicUnet <- function(encoder, n_classes, img_size, blur = FALSE,
                        blur_final = TRUE, self_attention = FALSE,
                        y_range = NULL, last_cross = TRUE, bottle = FALSE,
                        act_cls = nn()$ReLU, init = nn()$init$kaiming_normal_, norm_type = NULL) {

  args <- list(
    encoder = encoder,
    n_classes = n_classes,
    img_size = img_size,
    blur = blur,
    blur_final = blur_final,
    self_attention = self_attention,
    y_range = y_range,
    last_cross = last_cross,
    bottle = bottle,
    act_cls = act_cls,
    init = init,
    norm_type = norm_type
  )

  do.call(vision()$gan$DynamicUnet, args)

}



