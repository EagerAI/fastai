
#' @title Predict
#'
#' @description Prediction on `item`, fully decoded, loss function decoded and probabilities
#'
#' @param object the model
#' @param row row
#'
#' @export
predict.fastai.learner.Learner <- function(object, row) {

  #object$predict(reticulate::r_to_py(row)$iloc[0])[[3]]$numpy()
  # remove metric to obtain prediction
  object$metrics = object$metrics[0]

  cats = object$dls$vocab$items$items
  test_dl = object$dls$test_dl(row)
  predictions = object$get_preds(dl = test_dl, with_decoded = TRUE)
  res_cat = cats[predictions[[3]]$cpu()$numpy()+1]
  res_matrix = as.data.frame(predictions[[1]]$cpu()$numpy())
  names(res_matrix) = cats

  return(list(probabilities = res_matrix, labels = res_cat))

}
