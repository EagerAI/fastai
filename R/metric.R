

#' @title Flatten check
#'
#' @description Check that `out` and `targ` have the same number of elements and flatten them.
#'
#'
#' @param inp predictions
#' @param targ targets
#' @return tensor
#' @export
flatten_check <- function(inp, targ) {

  if(missing(inp) & missing(targ)) {
    metrics$flatten_check
  } else {
    metrics$flatten_check(
      inp = inp,
      targ = targ
    )
  }

}


#' @title AccumMetric
#'
#' @description Stores predictions and targets on CPU in accumulate to perform final calculations with `func`.
#'
#'
#' @param func function
#' @param dim_argmax dimension argmax
#' @param activation activation
#' @param thresh threshold point
#' @param to_np to matrix or not
#' @param invert_arg invert arguments
#' @param flatten flatten
#' @param ... additional arguments to pass
#' @return None
#' @export
AccumMetric <- function(func, dim_argmax = NULL, activation = "no",
                        thresh = NULL, to_np = FALSE,
                        invert_arg = FALSE, flatten = TRUE,
                        ...) {

  args <- list(
    func = func,
    dim_argmax = dim_argmax,
    activation = activation,
    thresh = thresh,
    to_np = to_np,
    invert_arg = invert_arg,
    flatten = flatten,
    ...
  )

  do.call(metrics$AccumMetric, args)

}

#' @title Skm to fastai
#'
#' @description Convert `func` from sklearn$metrics to a fastai metric
#'
#'
#' @param func function
#' @param is_class is classification or not
#' @param thresh threshold point
#' @param axis axis
#' @param activation activation
#' @param ... additional arguments to pass
#' @return None
#' @export
skm_to_fastai <- function(func, is_class = TRUE, thresh = NULL,
                          axis = -1, activation = NULL,
                          ...) {

  args <- list(
    func = func,
    is_class = is_class,
    thresh = thresh,
    axis = as.integer(axis),
    activation = activation,
    ...
  )

  do.call(metrics$skm_to_fastai, args)

}


#' @title Optim metric
#'
#' @description Replace metric `f` with a version that optimizes argument `argname`
#'
#'
#' @param f f
#' @param argname argname
#' @param bounds bounds
#' @param tol tol
#' @param do_neg do_neg
#' @param get_x get_x
#' @return None
#' @export
optim_metric <- function(f, argname, bounds,
                         tol = 0.01, do_neg = TRUE, get_x = FALSE) {

  args <- list(
    f = f,
    argname = argname,
    bounds = bounds,
    tol = tol,
    do_neg = do_neg,
    get_x = get_x
  )

  do.call(metrics$optim_metric, args)

}


#' @title Accuracy
#'
#' @description Compute accuracy with `targ` when `pred` is bs * n_classes
#'
#'
#' @param inp predictions
#' @param targ targets
#' @param axis axis
#' @return None
#' @export
accuracy <- function(inp, targ, axis = -1) {

  if(missing(inp) & missing(targ)) {
    metrics$accuracy
  } else {
    args <- list(
      inp = inp,
      targ = targ,
      axis = as.integer(axis)
    )
    do.call(metrics$accuracy, args)
  }

}


#' @title Top_k_accuracy
#'
#' @description Computes the Top-k accuracy (`targ` is in the top `k` predictions of `inp`)
#'
#'
#' @param inp predictions
#' @param targ targets
#' @param k k
#' @param axis axis
#' @return None
#' @export
top_k_accuracy <- function(inp, targ, k = 5, axis = -1) {


  if(missing(inp) & missing(targ)) {
    metrics$top_k_accuracy
  } else {
    args <- list(
      inp = inp,
      targ = targ,
      k = as.integer(k),
      axis = as.integer(axis)
    )

    do.call(metrics$top_k_accuracy, args)
  }

}

attr(top_k_accuracy, "py_function_name") <- 'top_k_accuracy'

#' @title APScoreBinary
#'
#' @description Average Precision for single-label binary classification problems
#'
#'
#' @param axis axis
#' @param average average
#' @param pos_label pos_label
#' @param sample_weight sample_weight
#' @return None
#' @export
APScoreBinary <- function(axis = -1, average = "macro", pos_label = 1, sample_weight = NULL) {

  args <- list(
    axis = as.integer(axis),
    average = average,
    pos_label = as.integer(pos_label),
    sample_weight = sample_weight
  )

  do.call(metrics$APScoreBinary, args)

}

#' @title BalancedAccuracy
#'
#' @description Balanced Accuracy for single-label binary classification problems
#'
#'
#' @param axis axis
#' @param sample_weight sample_weight
#' @param adjusted adjusted
#' @references None
#' @export
BalancedAccuracy <- function(axis = -1, sample_weight = NULL, adjusted = FALSE) {

  args <- list(
    axis = as.integer(axis),
    sample_weight = sample_weight,
    adjusted = adjusted
  )

  do.call(metrics$BalancedAccuracy, args)

}


#' @title BrierScore
#'
#' @description Brier score for single-label classification problems
#'
#'
#' @param axis axis
#' @param sample_weight sample_weight
#' @param pos_label pos_label
#' @return None
#' @export
BrierScore <- function(axis = -1, sample_weight = NULL, pos_label = NULL) {

  args <- list(
    axis = as.integer(axis),
    sample_weight = sample_weight,
    pos_label = pos_label
  )

  do.call(metrics$BrierScore, args)

}


#' @title CohenKappa
#'
#' @description Cohen kappa for single-label classification problems
#'
#'
#' @param axis axis
#' @param labels labels
#' @param weights weights
#' @param sample_weight sample_weight
#' @return None
#' @export
CohenKappa <- function(axis = -1, labels = NULL, weights = NULL, sample_weight = NULL) {

  args <- list(
    axis = as.integer(axis),
    labels = labels,
    weights = weights,
    sample_weight = sample_weight
  )

  do.call(metrics$CohenKappa, args)

}


#' @title F1Score
#'
#' @description F1 score for single-label classification problems
#'
#' @details
#'
#' @param axis axis
#' @param labels labels
#' @param pos_label pos_label
#' @param average average
#' @param sample_weight sample_weight
#' @return None
#' @export
F1Score <- function(axis = -1, labels = NULL, pos_label = 1, average = "binary", sample_weight = NULL) {

  args <- list(
    axis = as.integer(axis),
    labels = labels,
    pos_label = as.integer(pos_label),
    average = average,
    sample_weight = sample_weight
  )

  do.call(metrics$F1Score, args)

}

#' @title FBeta
#'
#' @description FBeta score with `beta` for single-label classification problems
#'
#'
#' @param beta beta
#' @param axis axis
#' @param labels labels
#' @param pos_label pos_label
#' @param average average
#' @param sample_weight sample_weight
#' @return None
#' @export
FBeta <- function(beta, axis = -1, labels = NULL, pos_label = 1, average = "binary", sample_weight = NULL) {


  if(missing(beta)) {
    metrics$FBeta
  } else {
    args <- list(
      beta = beta,
      axis = as.integer(axis),
      labels = labels,
      pos_label = as.integer(pos_label),
      average = average,
      sample_weight = sample_weight
    )

    do.call(metrics$FBeta, args)
  }

}

#' @title HammingLoss
#'
#' @description Hamming loss for single-label classification problems
#'
#'
#' @param axis axis
#' @param sample_weight sample_weight
#' @return None
#' @export
HammingLoss <- function(axis = -1, sample_weight = NULL) {

  args <- list(
    axis = as.integer(axis),
    sample_weight = sample_weight
  )

  do.call(metrics$HammingLoss, args)

}


#' @title Jaccard
#'
#' @description Jaccard score for single-label classification problems
#'
#'
#' @param axis axis
#' @param labels labels
#' @param pos_label pos_label
#' @param average average
#' @param sample_weight sample_weight
#' @return None
#' @export
Jaccard <- function(axis = -1, labels = NULL, pos_label = 1,
                    average = "binary", sample_weight = NULL) {

  args <- list(
    axis = as.integer(axis),
    labels = labels,
    pos_label = as.integer(pos_label),
    average = average,
    sample_weight = sample_weight
  )

  do.call(metrics$Jaccard, args)

}


#' @title Precision
#'
#' @description Precision for single-label classification problems
#'
#'
#' @param axis axis
#' @param labels labels
#' @param pos_label pos_label
#' @param average average
#' @param sample_weight sample_weight
#' @return None
#' @export
Precision <- function(axis = -1, labels = NULL, pos_label = 1,
                      average = "binary", sample_weight = NULL) {

  args <- list(
    axis = as.integer(axis),
    labels = labels,
    pos_label = as.integer(pos_label),
    average = average,
    sample_weight = sample_weight
  )

  do.call(metrics$Precision, args)

}


#' @title Recall
#'
#' @description Recall for single-label classification problems
#'
#'
#' @param axis axis
#' @param labels labels
#' @param pos_label pos_label
#' @param average average
#' @param sample_weight sample_weight
#' @return None
#' @export
Recall <- function(axis = -1, labels = NULL, pos_label = 1,
                   average = "binary", sample_weight = NULL) {

  args <- list(
    axis = as.integer(axis),
    labels = labels,
    pos_label = as.integer(pos_label),
    average = average,
    sample_weight = sample_weight
  )

  do.call(metrics$Recall, args)

}


#' @title RocAuc
#'
#' @description Area Under the Receiver Operating Characteristic
#' Curve for single-label multiclass classification problems
#'
#'
#' @param axis axis
#' @param average average
#' @param sample_weight sample_weight
#' @param max_fpr max_fpr
#' @param multi_class multi_class
#' @return None
#' @export
RocAuc <- function(axis = -1, average = "macro",
                   sample_weight = NULL, max_fpr = NULL, multi_class = "ovr") {

  args <- list(
    axis = as.integer(axis),
    average = average,
    sample_weight = sample_weight,
    max_fpr = max_fpr,
    multi_class = multi_class
  )

  do.call(metrics$RocAuc, args)

}


#' @title RocAucBinary
#'
#' @description Area Under the Receiver Operating Characteristic Curve for single-label binary classification problems
#'
#'
#' @param axis axis
#' @param average average
#' @param sample_weight sample_weight
#' @param max_fpr max_fpr
#' @param multi_class multi_class
#' @return None
#' @export
RocAucBinary <- function(axis = -1, average = "macro",
                         sample_weight = NULL, max_fpr = NULL, multi_class = "raise") {

  args <- list(
    axis = as.integer(axis),
    average = average,
    sample_weight = sample_weight,
    max_fpr = max_fpr,
    multi_class = multi_class
  )

  do.call(metrics$RocAucBinary, args)

}


#' @title MatthewsCorrCoef
#'
#' @description Matthews correlation coefficient for single-label classification problems
#' @param ... parameters to pass
#' @return None
#' @export
MatthewsCorrCoef <- function( ...) {

  args = list(...)

  if(length(args)>0){
    metrics$MatthewsCorrCoef(...)
  } else {
    metrics$MatthewsCorrCoef
  }

}


#' @title Perplexity
#'
#' @description Perplexity (exponential of cross-entropy loss) for Language Models
#'
#' @param ... parameters to pass
#' @return None
#' @export
preplexity = function(...) {

  args = list(...)

  if(length(args)>0) {
    do.call( metrics$perplexity,args)
  } else {
    metrics$perplexity
  }

}

#' @title Accuracy_multi
#'
#' @description Compute accuracy when `inp` and `targ` are the same size.
#'
#'
#' @param inp predictions
#' @param targ targets
#' @param thresh threshold point
#' @param sigmoid sigmoid
#' @return None
#' @export
accuracy_multi <- function(inp, targ, thresh = 0.5, sigmoid = TRUE) {

  if(missing(inp) & missing(targ)) {
    metrics$accuracy_multi
  } else {
    args <- list(
      inp = inp,
      targ = targ,
      thresh = thresh,
      sigmoid = sigmoid
    )
    do.call(metrics$accuracy_multi, args)
  }

}


#' @title APScoreMulti
#'
#' @description Average Precision for multi-label classification problems
#'
#'
#' @param sigmoid sigmoid
#' @param average average
#' @param pos_label pos_label
#' @param sample_weight sample_weight
#' @return None
#' @export
APScoreMulti <- function(sigmoid = TRUE, average = "macro",
                         pos_label = 1, sample_weight = NULL) {

  args <- list(
    sigmoid = sigmoid,
    average = average,
    pos_label = as.integer(pos_label),
    sample_weight = sample_weight
  )

  do.call(metrics$APScoreMulti, args)

}


#' @title BrierScoreMulti
#'
#' @description Brier score for multi-label classification problems
#'
#'
#' @param thresh thresh
#' @param sigmoid sigmoid
#' @param sample_weight sample_weight
#' @param pos_label pos_label
#' @return None
#' @export
BrierScoreMulti <- function(thresh = 0.5, sigmoid = TRUE,
                            sample_weight = NULL, pos_label = NULL) {

  metrics$BrierScoreMulti(
    thresh = thresh,
    sigmoid = sigmoid,
    sample_weight = sample_weight,
    pos_label = pos_label
  )

}



#' @title F1ScoreMulti
#'
#' @description F1 score for multi-label classification problems
#'
#' @param thresh thresh
#' @param sigmoid sigmoid
#' @param labels labels
#' @param pos_label pos_label
#' @param average average
#' @param sample_weight sample_weight
#' @return None
#' @export
F1ScoreMulti <- function(thresh = 0.5, sigmoid = TRUE, labels = NULL,
                         pos_label = 1, average = "macro",
                         sample_weight = NULL) {

  metrics$F1ScoreMulti(
    thresh = thresh,
    sigmoid = sigmoid,
    labels = labels,
    pos_label = as.integer(pos_label),
    average = average,
    sample_weight = sample_weight
  )

}


#' @title FBetaMulti
#'
#' @description FBeta score with `beta` for multi-label classification problems
#'
#'
#' @param beta beta
#' @param thresh thresh
#' @param sigmoid sigmoid
#' @param labels labels
#' @param pos_label pos_label
#' @param average average
#' @param sample_weight sample_weight
#' @return None
#' @export
FBetaMulti <- function(beta, thresh = 0.5, sigmoid = TRUE, labels = NULL,
                       pos_label = 1, average = "macro", sample_weight = NULL) {

  metrics$FBetaMulti(
    beta = beta,
    thresh = thresh,
    sigmoid = sigmoid,
    labels = labels,
    pos_label = as.integer(pos_label),
    average = average,
    sample_weight = sample_weight
  )

}





#' @title JaccardMulti
#'
#' @description Jaccard score for multi-label classification problems
#'
#' @param thresh thresh
#' @param sigmoid sigmoid
#' @param labels labels
#' @param pos_label pos_label
#' @param average average
#' @param sample_weight sample_weight
#' @return None
#' @export
JaccardMulti <- function(thresh = 0.5, sigmoid = TRUE,
                         labels = NULL, pos_label = 1,
                         average = "macro", sample_weight = NULL) {

  python_function_result <- metrics$JaccardMulti(
    thresh = thresh,
    sigmoid = sigmoid,
    labels = labels,
    pos_label = as.integer(pos_label),
    average = average,
    sample_weight = sample_weight
  )

}


#' @title MatthewsCorrCoefMulti
#'
#' @description Matthews correlation coefficient for multi-label classification problems
#'
#'
#' @param thresh thresh
#' @param sigmoid sigmoid
#' @param sample_weight sample_weight
#' @return None
#' @export
MatthewsCorrCoefMulti <- function(thresh = 0.5, sigmoid = TRUE, sample_weight = NULL) {

  metrics$MatthewsCorrCoefMulti(
    thresh = thresh,
    sigmoid = sigmoid,
    sample_weight = sample_weight
  )

}


#' @title PrecisionMulti
#'
#' @description Precision for multi-label classification problems
#'
#' @param thresh thresh
#' @param sigmoid sigmoid
#' @param labels labels
#' @param pos_label pos_label
#' @param average average
#' @param sample_weight sample_weight
#' @return None
#' @export
PrecisionMulti <- function(thresh = 0.5, sigmoid = TRUE, labels = NULL,
                           pos_label = 1, average = "macro",
                           sample_weight = NULL) {

  metrics$PrecisionMulti(
    thresh = thresh,
    sigmoid = sigmoid,
    labels = labels,
    pos_label = as.integer(pos_label),
    average = average,
    sample_weight = sample_weight
  )

}

#' @title RecallMulti
#'
#' @description Recall for multi-label classification problems
#'
#'
#' @param thresh thresh
#' @param sigmoid sigmoid
#' @param labels labels
#' @param pos_label pos_label
#' @param average average
#' @param sample_weight sample_weight
#' @return None
#' @export
RecallMulti <- function(thresh = 0.5, sigmoid = TRUE, labels = NULL,
                        pos_label = 1, average = "macro",
                        sample_weight = NULL) {

  metrics$RecallMulti(
    thresh = thresh,
    sigmoid = sigmoid,
    labels = labels,
    pos_label = as.integer(pos_label),
    average = average,
    sample_weight = sample_weight
  )

}

#' @title RocAucMulti
#'
#' @description Area Under the Receiver Operating Characteristic Curve for multi-label binary classification problems
#'
#'
#' @param sigmoid sigmoid
#' @param average average
#' @param sample_weight sample_weight
#' @param max_fpr max_fpr
#' @return None
#' @export
RocAucMulti <- function(sigmoid = TRUE, average = "macro",
                        sample_weight = NULL, max_fpr = NULL) {

  metrics$RocAucMulti(
    sigmoid = sigmoid,
    average = average,
    sample_weight = sample_weight,
    max_fpr = max_fpr
  )

}

#' @title MSE
#'
#' @description Mean squared error between `inp` and `targ`.
#'
#'
#' @param inp predictions
#' @param targ targets
#' @return None
#' @export
mse <- function(inp, targ) {

  if(missing(inp) & missing(targ)) {
    metrics$mse
  } else {
    metrics$mse(
      inp = inp,
      targ = targ
    )
  }

}


#' @title RMSE
#'
#' @description Root mean squared error
#'
#'
#' @param preds predictions
#' @param targs targets
#' @return None
#' @export
rmse <- function(preds, targs) {

  if(missing(preds) & missing(targs)) {
    metrics$rmse
  } else {
    metrics$rmse(
      preds = preds,
      targs = targs
    )
  }

}

#' @title MAE
#'
#' @description Mean absolute error between `inp` and `targ`.
#'
#'
#' @param inp predictions
#' @param targ targets
#' @return None
#' @export
mae <- function(inp, targ) {

  if(missing(inp) & missing(targ)) {
    metrics$mae
  } else {
    metrics$mae(
      inp = inp,
      targ = targ
    )
  }

}


#' @title MSLE
#'
#' @description Mean squared logarithmic error between `inp` and `targ`.
#'
#' @param inp predictions
#' @param targ targets
#' @return None
#' @export
msle <- function(inp, targ) {

  if(missing(inp) & missing(targ)) {
    metrics$msle
  } else {
    metrics$msle(
      inp = inp,
      targ = targ
    )
  }
}


#' @title Exp_rmspe
#'
#' @description Root mean square percentage error of the exponential of predictions and targets
#'
#'
#' @param preds predicitons
#' @param targs targets
#' @return None
#' @export
exp_rmspe <- function(preds, targs) {

  if(missing(preds) & missing(targs)) {
    metrics$exp_rmspe
  } else {
    metrics$exp_rmspe(
      preds = preds,
      targs = targs
    )
  }

}


#' @title Explained Variance
#'
#' @description Explained variance between predictions and targets
#'
#'
#' @param sample_weight sample_weight
#' @return None
#' @export
ExplainedVariance <- function(sample_weight = NULL) {

  metrics$ExplainedVariance(
    sample_weight = sample_weight
  )

}

#' @title R2Score
#'
#' @description R2 score between predictions and targets
#'
#'
#' @param sample_weight sample_weight
#' @return None
#' @export
R2Score <- function(sample_weight = NULL) {

  metrics$R2Score(
    sample_weight = sample_weight
  )

}


#' @title PearsonCorrCoef
#'
#' @description Pearson correlation coefficient for regression problem
#'
#'
#' @param dim_argmax dim_argmax
#' @param activation activation
#' @param thresh thresh
#' @param to_np to_np
#' @param invert_arg invert_arg
#' @param flatten flatten
#' @return None
#' @export
PearsonCorrCoef <- function(dim_argmax = NULL, activation = "no",
                            thresh = NULL, to_np = FALSE, invert_arg = FALSE, flatten = TRUE) {

  metrics$PearsonCorrCoef(
    dim_argmax = dim_argmax,
    activation = activation,
    thresh = thresh,
    to_np = to_np,
    invert_arg = invert_arg,
    flatten = flatten
  )

}


#' @title SpearmanCorrCoef
#'
#' @description Spearman correlation coefficient for regression problem
#'
#'
#' @param dim_argmax dim_argmax
#' @param axis axis
#' @param nan_policy nan_policy
#' @param activation activation
#' @param thresh thresh
#' @param to_np to_np
#' @param invert_arg invert_arg
#' @param flatten flatten
#' @return None
#' @export
SpearmanCorrCoef <- function(dim_argmax = NULL, axis = 0, nan_policy = "propagate",
                             activation = "no", thresh = NULL, to_np = FALSE,
                             invert_arg = FALSE, flatten = TRUE) {

  python_function_result <- metrics$SpearmanCorrCoef(
    dim_argmax = dim_argmax,
    axis = as.integer(axis),
    nan_policy = nan_policy,
    activation = activation,
    thresh = thresh,
    to_np = to_np,
    invert_arg = invert_arg,
    flatten = flatten
  )

}



#' @title Foreground accuracy
#'
#' @description Computes non-background accuracy for multiclass segmentation
#'
#'
#' @param inp predictions
#' @param targ targets
#' @param bkg_idx bkg_idx
#' @param axis axis
#' @return None
#' @export
foreground_acc <- function(inp, targ, bkg_idx = 0, axis = 1) {


  if(missing(inp) & missing(targ)) {
    metrics$foreground_acc
  } else {
    metrics$foreground_acc(
      inp = inp,
      targ = targ,
      bkg_idx = as.integer(bkg_idx),
      axis = as.integer(axis)
    )
  }

}

#' @title Dice coefficient
#'
#' @description Dice coefficient metric for binary target in segmentation
#'
#'
#' @param axis axis
#' @return None
#' @export
Dice <- function(axis = 1) {

  metrics$Dice(
    axis = as.integer(axis)
  )

}

#' @title JaccardCoeff
#'
#' @description Implementation of the Jaccard coefficient that is lighter in RAM
#'
#'
#' @param axis axis
#' @return None
#' @export
JaccardCoeff <- function(axis = 1) {

  metrics$JaccardCoeff(
    axis = as.integer(axis)
  )

}

#' @title CorpusBLEUMetric
#'
#' @description Blueprint for defining a metric
#'
#'
#' @param vocab_sz vocab_sz
#' @param axis axis
#' @return None
#' @export
CorpusBLEUMetric <- function(vocab_sz = 5000, axis = -1) {

  python_function_result <- metrics$CorpusBLEUMetric(
    vocab_sz = as.integer(vocab_sz),
    axis = as.integer(axis)
  )

}


#' @title LossMetric
#'
#' @description Create a metric from `loss_func.attr` named `nm`
#'
#' @param attr attr
#' @param nm nm
#' @return None
#' @export
LossMetric <- function(attr, nm = NULL) {

  metrics$LossMetric(
    attr = attr,
    nm = nm
  )

}
















