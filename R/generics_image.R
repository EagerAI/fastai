


#' @title Equal
#'
#' @param a tensor
#' @param b tensor
#' @return tensor
#' @rdname tensor_eq_img
#' @export
"==.fastai.torch_core.TensorImage" <- function(a, b) {
  a$eq(b)
}









