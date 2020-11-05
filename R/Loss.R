
#' @title MSELossFlat
#'
#' @description Flattens input and output, same as nn$MSELoss
#'
#' @param ... parameters to pass
#'
#' @return Loss object
#' @export
MSELossFlat = function(...) {
  args = list(...)

  do.call(vision()$all$MSELossFlat, args)
}


#' @title L1LossFlat
#'
#' @description Flattens input and output, same as nn$L1LossFlat
#' @param ... parameters to pass
#'
#' @return Loss object
#'
#' @export
L1LossFlat = function(...) {
  args = list(...)

  do.call(vision()$all$L1LossFlat, args)
}


#' @title BCELossFlat
#'
#' @description Flattens input and output, same as nn$BCELoss
#' @param ... parameters to pass
#'
#' @return Loss object
#'
#' @export
BCELossFlat = function(...) {
  args = list(...)

  do.call(vision()$all$BCELossFlat, args)
}



#' @title AvgLoss
#'
#' @description Flattens input and output, same as nn$AvgLoss
#' @param ... parameters to pass
#'
#' @return Loss object
#'
#' @export
AvgLoss = function(...) {
  args = list(...)

  do.call(vision()$all$AvgLoss, args)
}



#' @title BaseLoss
#'
#' @description Flattens input and output, same as nn$BaseLoss
#' @param ... parameters to pass
#' @return Loss object
#' @export
BaseLoss = function(...) {
  args = list(...)

  do.call(vision()$all$BaseLoss, args)
}




#' @title HammingLoss
#'
#' @description Hamming loss for single-label classification problems
#'
#'
#' @return Loss object
#'
#' @param axis axis
#' @param sample_weight sample weight
#'
#' @export
HammingLoss <- function(axis = -1, sample_weight = NULL) {

  vision()$all$HammingLoss(
    axis = as.integer(axis),
    sample_weight = sample_weight
  )

}


#' @title AdaptiveLoss
#'
#' @description Expand the `target` to match the `output` size before applying `crit`.
#'
#'
#' @return Loss object
#' @param crit critic
#'
#' @export
AdaptiveLoss <- function(crit) {

  vision()$gan$AdaptiveLoss(
    crit = crit
  )

}


#' @title HammingLossMulti
#'
#' @description Hamming loss for multi-label classification problems
#'
#' @return Loss object
#' @param thresh threshold
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
#' @return Loss object
#' @param beta beta, defaults to 0.98
#'
#' @export
AvgSmoothLoss <- function(beta = 0.98) {

  vision()$all$AvgSmoothLoss(
    beta = beta
  )

}


#' @title BCEWithLogitsLossFlat
#' @param ... parameters to pass
#'
#' @return Loss object
#' @export
BCEWithLogitsLossFlat = function(...) {
  args = list(...)

  do.call(vision()$all$BCEWithLogitsLossFlat, args)
}


#' @title LabelSmoothingCrossEntropy
#'
#' @description Same as `nn$Module`, but no need for subclasses to call `super()$__init__`
#'
#' @return Loss object
#'
#' @param eps epsilon
#' @param reduction reduction, defaults to mean
#'
#' @export
LabelSmoothingCrossEntropy <- function(eps = 0.1, reduction = "mean") {

  vision()$all$LabelSmoothingCrossEntropy(
    eps = eps,
    reduction = reduction
  )

}


#' @title LabelSmoothingCrossEntropyFlat
#'
#' @description Same as `nn$Module`, but no need for subclasses to call `super().__init__`
#'
#'
#' @param ... parameters to pass
#'
#' @return Loss object
#'
#' @export
LabelSmoothingCrossEntropyFlat <- function(...) {

  vision()$all$LabelSmoothingCrossEntropyFlat(
    ...
  )

}


#' @title CrossEntropyLossFlat
#'
#' @description Same as `nn$Module`, but no need for subclasses to call `super().__init__`
#'
#'
#' @param ... parameters to pass
#'
#' @return Loss object
#'
#' @export
CrossEntropyLossFlat <- function(...) {

  vision()$all$CrossEntropyLossFlat(
    ...
  )

}











