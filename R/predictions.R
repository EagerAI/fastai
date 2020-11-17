
#' @title Predict
#'
#' @description Prediction on `item`, fully decoded, loss function decoded and probabilities
#'
#' @param object the model
#' @param row row
#' @return data frame
#' @param ... additional arguments to pass
#' @export
predict.fastai.learner.Learner <- function(object, row, ...) {

  #object$predict(reticulate::r_to_py(row)$iloc[0])[[3]]$numpy()
  # remove metric to obtain prediction

  error_check = try(object$metrics[0],silent = TRUE)

  if(!inherits(error_check,'try-error')) {
    object$metrics <- error_check
  }

  test_dl = object$dls$test_dl(row)
  predictions = object$get_preds(dl = test_dl, with_decoded = TRUE)

  return(predictions)

}





