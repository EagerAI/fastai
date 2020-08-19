#' @title Adam
#'
#'
#' @param ... parameters to pass
#'
#' @export
Adam <- function(...) {
  args = list(...)

  if(is.null(unlist(args))) {
    tabular$Adam
  } else {
    do.call(tabular$Adam,args)
  }

}

attr(Adam ,"py_function_name") <- "Adam"
