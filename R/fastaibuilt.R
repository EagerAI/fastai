
#crap <- reticulate::import_from_path('crappify', path = 'fastaibuilt')

#' @title crappifier
#'
#'
#' @param path_lr path_lr
#' @param path_hr path_hr
#'
#' @export
crappifier <- function(path_lr, path_hr) {

  crap$crappifier(
    path_lr = path_lr,
    path_hr = path_hr
  )

}



#' @title RetinaNet
#'
#' @description Implements RetinaNet from https://arxiv.org/abs/1708.02002
#'
#' @details
#'
#' @param encoder encoder
#' @param n_classes n_classes
#' @param final_bias final_bias
#' @param chs chs
#' @param n_anchors n_anchors
#' @param flatten flatten
#'
#' @export
RetinaNet <- function(encoder, n_classes, final_bias = 0.0, chs = 256, n_anchors = 9, flatten = TRUE) {

  retinanet$RetinaNet(
    encoder = encoder,
    n_classes = n_classes,
    final_bias = final_bias,
    chs = as.integer(chs),
    n_anchors = as.integer(n_anchors),
    flatten = flatten
  )

}


#' @title RetinaNetFocalLoss
#'
#' @description Base class for all neural network modules.
#'
#' @details Your models should also subclass this class. Modules can also contain other Modules, allowing to nest them in
#' a tree structure. You can assign the submodules as regular attributes:: import torch.nn as nn import torch.nn.functional as F class Model(nn.Module): def __init__(self): super(Model, self).__init__() self.conv1 = nn.Conv2d(1, 20, 5) self.conv2 = nn.Conv2d(20, 20, 5) def forward(self, x): x = F.relu(self.conv1(x)) return F.relu(self.conv2(x)) Submodules assigned in this way will be registered, and will have their
#' parameters converted too when you call :meth:`to`, etc.
#'
#' @param gamma gamma
#' @param alpha alpha
#' @param pad_idx pad_idx
#' @param scales scales
#' @param ratios ratios
#' @param reg_loss reg_loss
#'
#' @export
RetinaNetFocalLoss <- function(gamma = 2.0, alpha = 0.25, pad_idx = 0,
                               scales = NULL, ratios = NULL,
                               reg_loss = nn$functional$smooth_l1_loss) {

  retinanet$RetinaNetFocalLoss(
    gamma = gamma,
    alpha = alpha,
    pad_idx = as.integer(pad_idx),
    scales = scales,
    ratios = ratios,
    reg_loss = reg_loss
  )

}




