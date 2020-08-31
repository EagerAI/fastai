

#' @title CSVLogger
#'
#' @description A `LearnerCallback` that saves history of metrics while training `learn` into CSV `filename`.
#'
#'
#' @param learn the model object
#' @param filename the name of the file
#' @param append append
#'
#' @export
CSVLogger <- function(learn, filename = "history", append = FALSE) {

  if(missing(learn)) {
    tabular$callbacks$CSVLogger
  } else {
    args <- list(
      learn = learn,
      filename = filename,
      append = append
    )

    do.call(tabular$callbacks$CSVLogger, args)
  }

}





