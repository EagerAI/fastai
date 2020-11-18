


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

  if(inherits(left, "torch.Tensor" ) & inherits(right, "torch.Tensor")) {
    left_expr <- paste(deparse(substitute(left),width.cutoff=500),
                      deparse(substitute(right),width.cutoff =500), sep = '<-')
    try(eval(parse(text = left_expr)), TRUE)
  } else if(!inherits(left, "torch.Tensor" ) & !inherits(right, "torch.Tensor")) {
    left_expr <- paste(deparse(substitute(left),width.cutoff=500),
                      deparse(substitute(right),width.cutoff=500), sep = '<-')
    try(eval(parse(text = left_expr)), TRUE)
  }  else {
    #deparse(substitute(x))

    cls = right
    lng = as.integer(length(left) - 1)

    if(inherits(cls,'numeric')) {
      left$put_(tensor(list(0L:lng)),tensor(right))
    } else if (inherits(cls,'integer')) {
      left$put_(tensor(list(0L:lng)),tensor(as.integer(right)))
    } else {
      stop("Pass R integer/numeric",call. = FALSE)
    }

    return(invisible(left))
  }
}

#' @title Modify tensor
#'
#' @param tensor torch tensor
#' @param slice dimension
#' @return tensor
#' @export
narrow = function(tensor, slice) {

  # tempdir
  temp = tempdir()
  torch()$save(tensor, paste(temp,'torch_a',sep = '/'))

  py_string = glue::glue("
import torch
a = torch.load('{temp}/torch_a')
a = a{slice}
torch.save(a, '{temp}/torch_a')
                          ")
  # fix windows
  py_string = gsub(py_string, replacement = '/',pattern = '\\',fixed=TRUE)
  reticulate::py_run_string(py_string)
  left = torch()$load(paste(temp,'torch_a',sep = '/'))
  return(left)
}


#' @title Fastai NN module
#'
#'
#' @param model_fn pass custom model function
#'
#' @return None
#' @export
nn_module = function(model_fn) {

  # if GPU is available move to gpu
  if(torch()$cuda$is_available()) {
    model <- Module_test()$RModel()$cuda()
  } else {
    model <- Module_test()$RModel()
  }
  r_model_call <- model_fn(model)
  model$`_r_call` <- r_model_call
  model
}

#' Operating system
#'
#'
#' @return vector
#' @export
os = function() {
  os = switch(Sys.info()[['sysname']],
              Windows= 'windows',
              Linux  = 'linux',
              Darwin = 'mac')
  os
}







