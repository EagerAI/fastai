#' Compares two tensors if equal
#'
#' This generic is approximately similar to \code{torch$eq(a, b)}, with the
#' difference that the generic returns a tensor of booleans instead of
#' a tensor of data type \code{torch$uint8}.
#'
#' @param a tensor
#' @param b tensor
#' @return A tensor of booleans, where False corresponds to 0, and 1 to True
#' in a tensor of data type \code{torch$bool}.
#'
#'
#' @export
"==.torch.Tensor" <- function(a, b) {
  a$eq(b)
}

#' Not equal
#'
#'
#' @name not_equal_to
#' @export
"!=.fastai.torch_core.TensorMask" <- function(a, b) {
  a$ne(b)
}


#' Mean of tensor
#'
#'
#'
#' @export
"mean.torch.Tensor" <- function(a) {
  a$mean()
}


#' @title Tensor to float
#'
#'
#'
#' @export
float <- function(tensor) {
  tensor$float()
}



#' @title To matrix
#'
#'
#' @export
to_matrix = function(obj) {
  if(inherits(obj,'pydicom.dataset.FileDataset')) {
    res = obj$as_dict()
    do.call(cbind,res)
  } else {
    fastai2$basics$tensor(obj)$cpu()$numpy()
  }
}

#' @title Print model
#'
#'
#' @export
print.fastai.learner.Learner = function(x, ...) {
  print(x$model)
}


