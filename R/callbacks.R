

#' @title CSVLogger
#'
#' @description A `LearnerCallback` that saves history of metrics while training `learn` into CSV `filename`.
#'
#'
#' @param learn learn
#' @param filename filename
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





