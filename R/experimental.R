


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
  }  else {
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

#' @title Modify tensor
#'
#' @param tensor torch tensor
#' @return tensor
#' @export
E = function(tensor) {
  # get slice
  string = deparse(substitute(a[,,,1]))

  string = gsub(" ", "", string, fixed = TRUE)

  string = gsub(",", ":,:", string)

  string = gsub('(.)\\1+', '\\1', string)

  # get tensor object
  a2 = sub("\\[.*", "", string)

  a = eval(parse(text = a2))

  # tempdir
  temp = tempdir()
  torch()$save(a, paste(temp,'torch_a',sep = '/'))

  reticulate::py_run_string(glue::glue("
import torch
a = torch.load('{temp}/torch_a')
a = {string}
torch.save(a, '{temp}/torch_a')
                          "))
  left = torch()$load(paste(temp,'torch_a',sep = '/'))
  return(left)
}





