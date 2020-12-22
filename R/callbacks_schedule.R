

#' @title SchedLin
#'
#' @description Linear schedule function from `start` to `end`
#'
#'
#' @param start start
#' @param end end
#' @return None
#' @export
SchedLin <- function(start, end) {

  fastai2$callback$schedule$SchedLin(
    start = start,
    end = end
  )

}


#' @title SchedCos
#'
#' @description Cosine schedule function from `start` to `end`
#'
#'
#' @param start start
#' @param end end
#' @return None
#' @export
SchedCos <- function(start, end) {

  fastai2$callback$schedule$SchedCos(
    start = start,
    end = end
  )

}


#' @title SchedNo
#'
#' @description Constant schedule function with `start` value
#'
#'
#' @param start start
#' @param end end
#' @return None
#' @export
SchedNo <- function(start, end) {

  fastai2$callback$schedule$SchedNo(
    start = start,
    end = end
  )

}

#' @title SchedExp
#'
#' @description Exponential schedule function from `start` to `end`
#'
#'
#' @param start start
#' @param end end
#' @return None
#' @export
SchedExp <- function(start, end) {

  fastai2$callback$schedule$SchedExp(
    start = start,
    end = end
  )

}


#' @title SchedPoly
#'
#' @description Polynomial schedule (of `power`) function from `start` to `end`
#'
#'
#' @param start start
#' @param end end
#' @param power power
#' @return None
#' @export
SchedPoly <- function(start, end, power) {

  fastai2$callback$schedule$SchedPoly(
    start = start,
    end = end,
    power = power
  )

}



#' @title ParamScheduler
#'
#' @description Schedule hyper-parameters according to `scheds`
#'
#'
#' @param scheds scheds
#' @return None
#' @export
ParamScheduler <- function(scheds) {

  fastai2$callback$schedule$ParamScheduler(
    scheds = scheds
  )

}












