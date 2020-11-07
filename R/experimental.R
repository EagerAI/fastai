


#' @title Fastai assignment
#'
#' @description The assignment has to be used for safe modification of the values inside tensors/layers
#'
#' @param left left side object
#' @param right right side object
#' @return None
#'
#' @export
`%f%` <- function(left, right) {

  if(!inherits(left, "torch.Tensor" )) {
    if(inherits(right, "integer")) {
      left_expr = paste(deparse(substitute(left)), paste(right,"L",sep = ''), sep = '<-')
    } else {
      left_expr = paste(deparse(substitute(left)), right, sep = '<-')
    }
    return(try(eval(parse(text = left_expr)), TRUE))
  } else {
    #deparse(substitute(x))

    cls = right
    lng = as.integer(length(left) - 1)

    if(inherits(cls,'numeric')) {
      left$put_(tensor(list(0L:lng)),tensor(right))
    } else if (inherits(cls,'integer')) {
      left$put_(tensor(list(0L:lng)),tensor(as.integer(right)))
    } else if(inherits(cls, 'torch.Tensor')) {
      if(right$dtype$is_floating_point != left$dtype$is_floating_point) {
        stop('Cannot assign integer to numeric tensor', call. = FALSE)
      } else {
        left$put_(tensor(list(0L:lng)), right)
      }
    }
    return(invisible(left))
  }

}




