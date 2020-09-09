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


#' @title RAdam
#'
#'
#' @param ... parameters to pass
#'
#' @export
RAdam <- function(...) {
  args = list(...)

  if(is.null(unlist(args))) {
    vision$all$RAdam
  } else {
    do.call(vision$all$RAdam, args)
  }

}

attr(RAdam ,"py_function_name") <- "RAdam"

#' @title SGD
#'
#'
#' @param ... parameters to pass
#'
#' @export
SGD <- function(...) {
  args = list(...)

  if(is.null(unlist(args))) {
    vision$all$SGD
  } else {
    do.call(vision$all$SGD, args)
  }

}

attr(SGD ,"py_function_name") <- "SGD"


#' @title RMSProp
#'
#'
#' @param ... parameters to pass
#'
#' @export
RMSProp <- function(...) {
  args = list(...)

  if(is.null(unlist(args))) {
    vision$all$RMSProp
  } else {
    do.call(vision$all$RMSProp, args)
  }

}

attr(RMSProp ,"py_function_name") <- "RMSProp"


#' @title QHAdam
#'
#'
#' @param ... parameters to pass
#'
#' @export
QHAdam <- function(...) {
  args = list(...)

  if(is.null(unlist(args))) {
    vision$all$QHAdam
  } else {
    do.call(vision$all$QHAdam, args)
  }

}

attr(QHAdam ,"py_function_name") <- "QHAdam"


#' @title Larc
#'
#'
#' @param ... parameters to pass
#'
#' @export
Larc <- function(...) {
  args = list(...)

  if(is.null(unlist(args))) {
    vision$all$Larc
  } else {
    do.call(vision$all$Larc, args)
  }

}

#' @title Lamb
#'
#'
#' @param ... parameters to pass
#'
#' @export
Lamb <- function(...) {
  args = list(...)

  if(is.null(unlist(args))) {
    vision$all$Lamb
  } else {
    do.call(vision$all$Lamb, args)
  }

}



#' @title Lookahead
#'
#'
#' @param ... parameters to pass
#'
#' @export
Lookahead <- function(...) {
  args = list(...)

  if(is.null(unlist(args))) {
    vision$all$Lookahead
  } else {
    do.call(vision$all$Lookahead, args)
  }

}


#' @title OptimWrapper
#'
#'
#' @param ... parameters to pass
#'
#' @export
OptimWrapper <- function(...) {
  args = list(...)

  if(is.null(unlist(args))) {
    vision$all$OptimWrapper
  } else {
    do.call(vision$all$OptimWrapper, args)
  }

}


#' @title Optimizer
#'
#'
#' @param ... parameters to pass
#'
#' @export
Optimizer <- function(...) {
  args = list(...)

  if(is.null(unlist(args))) {
    vision$all$Optimizer
  } else {
    do.call(vision$all$Optimizer, args)
  }

}









