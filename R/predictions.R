
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

  error_check = try(object$metrics[0],silent = TRUE, outFile = 'error')

  if(!inherits(error_check,'try-error')) {
    object$metrics = error_check
  }

  cats = try(object$dls$vocab$items$items, silent = TRUE)

  if(!inherits(cats,'try-error')) {
    test_dl = object$dls$test_dl(row)
    predictions = object$get_preds(dl = test_dl, with_decoded = TRUE)
    res_cat = cats[predictions[[3]]$cpu()$numpy()+1]
    res_matrix = as.data.frame(predictions[[1]]$cpu()$numpy())
    names(res_matrix) = cats
    output = list(probabilities = res_matrix, labels = res_cat)
  } else {
    test_dl = object$dls$test_dl(row)
    predictions = object$get_preds(dl = test_dl, with_decoded = TRUE)
    output = predictions[[1]]
  }

  return(output)

}
