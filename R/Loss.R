
#' @title MSELossFlat
#'
#' @description Flattens input and output, same as nn$MSELoss
#'
#'
#' @export
MSELossFlat = function(...) {
  args = list(...)

  do.call(vision$all$MSELossFlat, args)
}


#' @title L1LossFlat
#'
#' @description Flattens input and output, same as nn$L1LossFlat
#'
#'
#' @export
L1LossFlat = function(...) {
  args = list(...)

  do.call(vision$all$L1LossFlat, args)
}


#' @title BCELossFlat
#'
#' @description Flattens input and output, same as nn$BCELoss
#'
#'
#' @export
BCELossFlat = function(...) {
  args = list(...)

  do.call(vision$all$BCELossFlat, args)
}



#' @title AvgLoss
#'
#' @description Flattens input and output, same as nn$AvgLoss
#'
#'
#' @export
AvgLoss = function(...) {
  args = list(...)

  do.call(vision$all$AvgLoss, args)
}



#' @title BaseLoss
#'
#' @description Flattens input and output, same as nn$BaseLoss
#'
#'
#' @export
BaseLoss = function(...) {
  args = list(...)

  do.call(vision$all$BaseLoss, args)
}




#' @title HammingLoss
#'
#' @description Hamming loss for single-label classification problems
#'
#' @details
#'
#' @param axis axis
#' @param sample_weight sample_weight
#'
#' @export
HammingLoss <- function(axis = -1, sample_weight = NULL) {

  vision$all$HammingLoss(
    axis = as.integer(axis),
    sample_weight = sample_weight
  )

}


#' @title AdaptiveLoss
#'
#' @description Expand the `target` to match the `output` size before applying `crit`.
#'
#'
#' @param crit crit
#'
#' @export
AdaptiveLoss <- function(crit) {

  vision$gan$AdaptiveLoss(
    crit = crit
  )

}


#' @title HammingLossMulti
#'
#' @description Hamming loss for multi-label classification problems
#'
#'
#' @param thresh thresh
#' @param sigmoid sigmoid
#' @param labels labels
#' @param sample_weight sample_weight
#'
#' @export
HammingLossMulti <- function(thresh = 0.5, sigmoid = TRUE, labels = NULL, sample_weight = NULL) {

  fastai2$vision$all$HammingLossMulti(
    thresh = thresh,
    sigmoid = sigmoid,
    labels = labels,
    sample_weight = sample_weight
  )

}



#' @title AvgSmoothLoss
#'
#' @description Smooth average of the losses (exponentially weighted with `beta`)
#'
#'
#' @param beta beta
#'
#' @export
AvgSmoothLoss <- function(beta = 0.98) {

  vision$all$AvgSmoothLoss(
    beta = beta
  )

}


#' BCEWithLogitsLossFlat
#'
#' @export
BCEWithLogitsLossFlat = function(...) {
  args = list(...)

  do.call(vision$all$BCEWithLogitsLossFlat, args)
}


#' @title LabelSmoothingCrossEntropy
#'
#' @description Same as `nn.Module`, but no need for subclasses to call `super().__init__`
#'
#'
#' @param eps eps
#' @param reduction reduction
#'
#' @export
LabelSmoothingCrossEntropy <- function(eps = 0.1, reduction = "mean") {

  vision$all$LabelSmoothingCrossEntropy(
    eps = eps,
    reduction = reduction
  )

}


#' @title LabelSmoothingCrossEntropyFlat
#'
#' @description Same as `nn.Module`, but no need for subclasses to call `super().__init__`
#'
#'
#' @param ... additional parameters to pass
#'
#' @export
LabelSmoothingCrossEntropyFlat <- function(...) {

  vision$all$LabelSmoothingCrossEntropyFlat(
    ...
  )

}


#' @title CrossEntropyLossFlat
#'
#' @description Same as `nn.Module`, but no need for subclasses to call `super().__init__`
#'
#'
#' @param ... additional parameters to pass
#'
#' @export
CrossEntropyLossFlat <- function(...) {

  vision$all$CrossEntropyLossFlat(
    ...
  )

}











