
#' @title Fill Missing
#'
#' @description Fill the missing values in continuous columns.
#'
#' @param cat_names The names of the categorical variables
#' @param cont_names The names of the continuous variables
#' @param fill_strategy The strategy of filling
#' @param add_col add_col
#' @param fill_val fill_val
#' @return None
#' @export
FillMissing <- function(cat_names, cont_names, fill_strategy = FillStrategy_MEDIAN(), add_col = TRUE, fill_val = 0.0) {


  if (missing(cat_names) & missing(cont_names)) {

    tabular$FillMissing
  } else {
    args <- list(
      cat_names = cat_names,
      cont_names = cont_names,
      fill_strategy = fill_strategy,
      add_col = add_col,
      fill_val = fill_val
    )

    do.call(tabular$FillMissing, args)
  }

}



#' @title Normalize
#'
#' @description Normalize the continuous variables.
#'
#'
#' @param cat_names cat_names
#' @param cont_names cont_names
#' @return None
#' @export
Normalize <- function(cat_names, cont_names) {

  if(missing(cat_names) & missing(cont_names)) {
    tabular$Normalize
  } else {
    args <- list(
      cat_names = cat_names,
      cont_names = cont_names
    )

    do.call(tabular$Normalize, args)
  }

}


#' @title Categorify
#'
#' @description Transform the categorical variables to that type.
#'
#'
#' @param cat_names The names of the categorical variables
#' @param cont_names The names of the continuous variables
#' @return None
#' @export
Categorify <- function(cat_names, cont_names) {

  if(missing(cat_names) & missing(cont_names)) {
    tabular$Categorify
  } else {
    args <- list(
      cat_names = cat_names,
      cont_names = cont_names
    )

    do.call(tabular$Categorify, args)
  }

}


#' @title MEDIAN
#'
#' @description An enumeration.
#' @return None
#' @export
FillStrategy_MEDIAN <- function() {
  tabular$FillStrategy$MEDIAN
}



#' @title COMMON
#'
#' @description An enumeration.
#' @return None
#' @export
FillStrategy_COMMON <- function() {
  tabular$FillStrategy$COMMON
}


#' @title CONSTANT
#'
#' @description An enumeration.
#' @return None
#' @export
FillStrategy_CONSTANT <- function() {
  tabular$FillStrategy$CONSTANT
}



#' @title Add datepart
#'
#' @description Helper function that adds columns relevant to a date in the column `field_name` of `df`.
#'
#'
#' @param df df
#' @param field_name field_name
#' @param prefix prefix
#' @param drop drop
#' @param time time
#' @return data frame
#' @export
add_datepart <- function(df, field_name, prefix = NULL, drop = TRUE, time = FALSE) {

  args <- list(
    df = df,
    field_name = field_name,
    prefix = prefix,
    drop = drop,
    time = time
  )

  do.call(tabular$add_datepart, args)

}

#' @title Add cyclic datepart
#'
#' @description Helper function that adds trigonometric date/time features to a date in the column `field_name` of `df`.
#'
#'
#' @param df df
#' @param field_name field_name
#' @param prefix prefix
#' @param drop drop
#' @param time time
#' @param add_linear add_linear
#' @return data frame
#' @export
add_cyclic_datepart <- function(df, field_name, prefix = NULL, drop = TRUE, time = FALSE, add_linear = FALSE) {

  args <- list(
    df = df,
    field_name = field_name,
    prefix = prefix,
    drop = drop,
    time = time,
    add_linear = add_linear
  )

  do.call(tabular$add_cyclic_datepart, args)

}


#' @title Predict
#'
#' @description Prediction on `item`, fully decoded, loss function decoded and probabilities
#' @importFrom reticulate r_to_py
#'
#' @param object the model
#' @param row row
#' @return data frame
#' @param ... additional arguments to pass
#' @export
predict.fastai.tabular.learner.TabularLearner <- function(object, ..., row) {

  #object$predict(reticulate::r_to_py(row)$iloc[0])[[3]]$numpy()
  # remove metric to obtain prediction
  object$metrics = object$metrics[0]

  test_dl = object$dls$test_dl(row)
  predictions = object$get_preds(dl = test_dl, with_decoded = TRUE)
  res = as.data.frame(predictions[[1]]$cpu()$numpy())
  class = predictions[[3]]$cpu()$numpy()
  output = try(object$dls$vocab$items$items, silent = TRUE)

  if(inherits(output,'try-error')) {
    names(res) = object$dl$y_names$items
  } else {
    names(res) = object$dls$vocab$items$items
    res = cbind(res,class)
  }
  res
}
