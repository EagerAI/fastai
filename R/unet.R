#' @title Unet_config
#'
#' @description Convenience function to easily create a config for `DynamicUnet`
#'
#'
#' @param blur blur
#' @param blur_final blur_final
#' @param self_attention self_attention
#' @param y_range y_range
#' @param last_cross last_cross
#' @param bottle bottle
#' @param act_cls act_cls
#' @param init init
#' @param norm_type norm_type
#'
#' @export
unet_config <- function(blur = FALSE, blur_final = TRUE, self_attention = FALSE,
                        y_range = NULL, last_cross = TRUE, bottle = FALSE,
                        act_cls = nn$ReLU, init = nn$init$kaiming_normal_, norm_type = NULL) {

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

  do.call(vision$gan$unet_config, args)

}




#' @title Unet_learner
#'
#' @description Build a unet learner from `dls` and `arch`
#'
#' @param dls dls
#' @param arch arch
#' @param loss_func loss_func
#' @param pretrained pretrained
#' @param cut cut
#' @param splitter splitter
#' @param config config
#' @param n_in n_in
#' @param n_out n_out
#' @param normalize normalize
#' @param opt_func opt_func
#' @param lr lr
#' @param cbs cbs
#' @param metrics metrics
#' @param path path
#' @param model_dir model_dir
#' @param wd wd
#' @param wd_bn_bias wd_bn_bias
#' @param train_bn train_bn
#' @param moms moms
#'
#' @export
unet_learner <- function(dls, arch, loss_func = NULL, pretrained = TRUE,
                         cut = NULL, splitter = NULL, config = NULL, n_in = 3,
                         n_out = NULL, normalize = TRUE, opt_func = Adam(), lr = 0.001,
                         cbs = NULL, metrics = NULL, path = NULL, model_dir = "models",
                         wd = NULL, wd_bn_bias = FALSE, train_bn = TRUE,
                         moms = list(0.95, 0.85, 0.95)) {

  args <- list(
    dls = dls,
    arch = arch,
    loss_func = loss_func,
    pretrained = pretrained,
    cut = cut,
    splitter = splitter,
    config = config,
    n_in = as.integer(n_in),
    n_out = n_out,
    normalize = normalize,
    opt_func = opt_func,
    lr = lr,
    cbs = cbs,
    metrics = metrics,
    path = path,
    model_dir = model_dir,
    wd = wd,
    wd_bn_bias = wd_bn_bias,
    train_bn = train_bn,
    moms = moms
  )

  do.call(vision$gan$unet_learner, args)

}


#' @title UnetBlock
#'
#' @description A quasi-UNet block, using `PixelShuffle_ICNR upsampling`.
#'
#'
#' @param up_in_c up_in_c
#' @param x_in_c x_in_c
#' @param hook hook
#' @param final_div final_div
#' @param blur blur
#' @param act_cls act_cls
#' @param self_attention self_attention
#' @param init init
#' @param norm_type norm_type
#' @param ks ks
#' @param stride stride
#' @param padding padding
#' @param bias bias
#' @param ndim ndim
#' @param bn_1st bn_1st
#' @param transpose transpose
#' @param xtra xtra
#' @param bias_std bias_std
#' @param dilation dilation
#' @param groups groups
#' @param padding_mode padding_mode
#'
#' @export
UnetBlock <- function(up_in_c, x_in_c, hook, final_div = TRUE,
                      blur = FALSE, act_cls = nn$ReLU, self_attention = FALSE,
                      init = nn$init$kaiming_normal_, norm_type = NULL, ks = 3,
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

  do.call(vision$gan$UnetBlock, args)

}


#' @title DynamicUnet
#'
#' @description Create a U-Net from a given architecture.
#'
#'
#' @param encoder encoder
#' @param n_classes n_classes
#' @param img_size img_size
#' @param blur blur
#' @param blur_final blur_final
#' @param self_attention self_attention
#' @param y_range y_range
#' @param last_cross last_cross
#' @param bottle bottle
#' @param act_cls act_cls
#' @param init init
#' @param norm_type norm_type
#'
#' @export
DynamicUnet <- function(encoder, n_classes, img_size, blur = FALSE,
                        blur_final = TRUE, self_attention = FALSE,
                        y_range = NULL, last_cross = TRUE, bottle = FALSE,
                        act_cls = nn$ReLU, init = nn$init$kaiming_normal_, norm_type = NULL) {

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

  do.call(vision$gan$DynamicUnet, args)

}







