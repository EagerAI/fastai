
#' @title Accuracy
#'
#' @description Computes accuracy with `targs` when `input` is bs * n_classes.
#'
#' @param input input
#' @param targs targs
#'
#' @export
accuracy <- function(input, targs) {


  if(missing(input) & missing(targs)) {

    tabular$accuracy
  } else {
    args <- list(
      input = input,
      targs = targs
    )

    do.call(tabular$accuracy, args)
  }

}


#' @title Fill Missing
#'
#' @description Fill the missing values in continuous columns.
#'
#' @param cat_names cat_names
#' @param cont_names cont_names
#' @param fill_strategy fill_strategy
#' @param add_col add_col
#' @param fill_val fill_val
#'
#' @export
FillMissing <- function(cat_names, cont_names, fill_strategy = MEDIAN(), add_col = TRUE, fill_val = 0.0) {


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

#' @title MEDIAN
#'
#' @description An enumeration.
#'
#' @export
MEDIAN <- function() {
  tabular$FillStrategy$MEDIAN
}


#' @title Normalize
#'
#' @description Normalize the continuous variables.
#'
#'
#' @param cat_names cat_names
#' @param cont_names cont_names
#'
#' @export
Normalize <- function(cat_names, cont_names) {

  if(missing(cat_names) & missing(cont_names)) {
    tabular$Categorify
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
#' @param cat_names cat_names
#' @param cont_names cont_names
#'
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


#' @title Tabular List from dataframe
#'
#' @description Get the list of inputs in the `col` of `path/csv_name`.
#'
#' @param cls cls
#' @param df df
#' @param cat_names cat_names
#' @param cont_names cont_names
#' @param procs procs
#'
#' @export
tabular_TabularList_from_df <- function(df, cat_names = NULL, cont_names = NULL, procs = NULL) {

  args <- list(
    df = df,
    cat_names = cat_names,
    cont_names = cont_names,
    procs = procs
  )
  do.call(tabular$TabularList$from_df, args)

}


#' @title Tabular learner
#'
#' @description Get a `Learner` using `data`, with `metrics`,
#' including a `TabularModel` created using the remaining params.
#'
#'
#' @param data data
#' @param layers layers
#' @param emb_szs emb_szs
#' @param metrics metrics
#' @param ps ps
#' @param emb_drop emb_drop
#' @param y_range y_range
#' @param use_bn use_bn
#'
#' @export
tabular_learner <- function(data, layers, emb_szs = NULL, metrics = NULL,
                            path = NULL,
                            ps = NULL, emb_drop = 0.0, y_range = NULL, use_bn = TRUE) {

  args <- list(
    data = data,
    layers = layers,
    emb_szs = emb_szs,
    metrics = metrics,
    path = path,
    ps = ps,
    emb_drop = emb_drop,
    y_range = y_range,
    use_bn = use_bn
  )

  do.call(tabular$tabular_learner, args)

}


#' @title Fit
#' @description Fit the model on this learner with `lr` learning rate, `wd` weight decay for `epochs` with `callbacks`.
#'
#' @param epochs epochs
#' @param lr lr
#' @param wd wd
#' @param callbacks callbacks
#'
#' @export
fit.fastai.basic_train.Learner <- function(object, epochs, lr = 1e-2, wd = NULL, callbacks = NULL) {

  args <- list(
    epochs = epochs,
    lr = lr,
    wd = wd,
    callbacks = callbacks
  )
  # fit the model
  history <- do.call(object$fit, args)
}



