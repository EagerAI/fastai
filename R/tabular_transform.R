
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





