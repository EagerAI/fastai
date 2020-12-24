

#' @title MixHandler
#'
#' @description A handler class for implementing `MixUp` style scheduling
#'
#'
#' @param alpha alpha
#' @return None
#'
#' @export
MixHandler <- function(alpha = 0.5) {

  fastai2$callback$mixup$MixHandler(
    alpha = alpha
  )

}


#' @title MixUp
#'
#' @description Implementation of https://arxiv.org/abs/1710.09412
#'
#'
#' @param alpha alpha
#' @return None
#'
#' @export
MixUp <- function(alpha = 0.4) {

  fastai2$callback$mixup$MixUp(
    alpha = alpha
  )

}


#' @title CutMix
#'
#' @description Implementation of `https://arxiv.org/abs/1905.04899`
#'
#'
#' @param alpha alpha
#' @return None
#'
#' @export
CutMix <- function(alpha = 1.0) {

  fastai2$callback$mixup$CutMix(
    alpha = alpha
  )

}



#' @title MCDropoutCallback
#'
#' @description Turns on dropout during inference, allowing you to call
#' Learner$get_preds multiple times to approximate your model
#' uncertainty using Monte Carlo Dropout. https://arxiv.org/pdf/1506.02142.pdf
#'
#' @param ... arguments to pass
#' @return None
#'
#' @export
MCDropoutCallback = function(...) {

  fastai2$callback$preds$MCDropoutCallback(...)

}


#' @title ModelResetter
#'
#' @description Callback that resets the model at each validation/training step
#'
#' @param ... arguments to pass
#' @return None
#'
#' @export
ModelResetter = function(...) {

  fastai2$callback$rnn$ModelResetter(...)

}


#' @title RNNRegularizer
#'
#' @description `Callback` that adds AR and TAR regularization in RNN training
#'
#'
#' @param alpha alpha
#' @param beta beta
#' @return None
#' @export
RNNRegularizer <- function(alpha = 0.0, beta = 0.0) {

  fastai2$callback$rnn$RNNRegularizer(
    alpha = alpha,
    beta = beta
  )

}










