
#crap <- reticulate::import_from_path('crappify', path = 'fastaibuilt')

#' @title Crappifier
#'
#'
#' @param path_lr path from (origin)
#' @param path_hr path to (destination)
#' @return None
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
#'
#' @param ... arguments to pass
#' @return model
#' @export
RetinaNet <- function(...) {

  args = list(
    ...
  )

  if(!is.null(args[['n_classes']])) {
    args[['n_classes']] = as.integer(args[['n_classes']])
  }

  if(!is.null(args[['chs']])) {
    args[['chs']] = as.integer(args[['chs']])
  }

  if(!is.null(args[['n_anchors']])) {
    args[['n_anchors']] = as.integer(args[['n_anchors']])
  }

  do.call(retinanet$RetinaNet, args)
}


#' @title RetinaNetFocalLoss
#'
#' @description Base class for all neural network modules.
#'
#' @details Your models should also subclass this class. Modules can also contain other Modules, allowing to nest them in
#' a tree structure. You can assign the submodules as regular attributes:: import torch.nn as nn import torch.nn.functional as F class Model(nn.Module): def __init__(self): super(Model, self).__init__() self.conv1 = nn.Conv2d(1, 20, 5) self.conv2 = nn.Conv2d(20, 20, 5) def forward(self, x): x = F.relu(self.conv1(x)) return F.relu(self.conv2(x)) Submodules assigned in this way will be registered, and will have their
#' parameters converted too when you call :meth:`to`, etc.
#'
#' @param ... parameters to pass
#' @return None
#' @export
RetinaNetFocalLoss <- function(...) {

  args = list(
    ...
  )

  if(!is.null(args[['pad_idx']])) {
    args[['pad_idx']] = as.integer(args[['pad_idx']])
  }

  do.call(retinanet$RetinaNetFocalLoss, args)

}




