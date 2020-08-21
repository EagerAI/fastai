#' @title Adam
#'
#'
#' @param ... parameters to pass
#'
#' @export
Adam <- function(...) {
  args = list(...)

  if(is.null(unlist(args))) {
    vision$all$Adam
  } else {
    do.call(vision$all$Adam, args)
  }

}

attr(Adam ,"py_function_name") <- "Adam"
