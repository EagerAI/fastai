
#' @title Fill Missing
#'
#' @description Fill the missing values in continuous columns.
#'
#' @param cat_names The names of the categorical variables
#' @param cont_names The names of the continuous variables
#' @param fill_strategy The strategy of filling
#' @param add_col add_col
#' @param fill_val fill_val
#'
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
#'
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


#' @title MEDIAN
#'
#' @description An enumeration.
#'
#' @export
FillStrategy_MEDIAN <- function() {
  tabular$FillStrategy$MEDIAN
}



#' @title COMMON
#'
#' @description An enumeration.
#'
#' @export
FillStrategy_COMMON <- function() {
  tabular$FillStrategy$COMMON
}


#' @title CONSTANT
#'
#' @description An enumeration.
#'
#' @export
FillStrategy_CONSTANT <- function() {
  tabular$FillStrategy$CONSTANT
}






#' @title Apply tabular transformation
#' @importFrom data.table := set
#' @description An enumeration.
#'
#' @export
tabular_apply <- function(object, DT) {

  # fill
  islist = tryCatch({identical(class(object$na_dict), "list")
  }, error = function(e){FALSE})

  # norm
  islist2 = tryCatch({identical(class(object$means), "list")
  }, error = function(e){FALSE})

  if(!islist & !islist2)
    tryCatch({object(DT)}, error = function(e){FALSE})

  # fill
  islist = tryCatch({identical(class(object$na_dict), "list")
  }, error = function(e){FALSE})

  # norm
  islist2 = tryCatch({identical(class(object$means), "list")
  }, error = function(e){FALSE})

  if(islist) {
    for (j in names(object$na_dict))
      set(DT,which(is.na(DT[[j]])), j, object$na_dict[j])
  } else {
    DT[, c(names(object$means)) := lapply(.SD, function(x) (x - mean(x))/sd(x) ), .SDcols = names(object$means)]
  }

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
#'
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
#'
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

#' @title Show_xys
#'
#' @description Show the `xs` (inputs) and `ys` (targets).
#'
#' @details
#'
#' @param xs xs
#' @param ys ys
#'
#' @export
tabular_TabularList_show_xys <- function(xs, ys) {

  tabular$TabularList$show_xys(
    xs = xs,
    ys = ys
  )

}

#' @title show_xyzs
#'
#' @description Show `xs` (inputs), `ys` (targets) and `zs` (predictions).
#'
#' @details
#'
#' @param xs xs
#' @param ys ys
#' @param zs zs
#'
#' @export
tabular_TabularList_show_xyzs <- function(xs, ys, zs) {

  tabular$TabularList$show_xyzs(
    xs = xs,
    ys = ys,
    zs = zs
  )

}


#' @title Predict
#'
#' @description Prediction on `item`, fully decoded, loss function decoded and probabilities
#' @importFrom reticulate r_to_py
#'
#' @param object the model
#' @param row row
#'
#' @export
predict.fastai.tabular.learner.TabularLearner <- function(object, row) {

  object$predict(reticulate::r_to_py(row)$iloc[0])[[3]]$numpy()

}
