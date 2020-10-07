
#' @title Equal
#'
#' @param a tensor
#' @param b tensor
#' @return tensor
#' @export
"==.fastai.torch_core.TensorMask" <- function(a, b) {
  a$eq(b)
}


#' @title Pow
#'
#' @param a tensor
#' @param b tensor
#' @return tensor
#' @export
"^.fastai.torch_core.TensorMask" <- function(a, b) {
  a$pow(b)
}

#' @title Not equal
#'
#' @param a tensor
#' @param b tensor
#' @return tensor
#' @name not_equal_to_mask_
#' @export
"!=.fastai.torch_core.TensorMask" <- function(a, b) {
  a$ne(b)
}


#' @title Greater or equal
#'
#' @param a tensor
#' @param b tensor
#' @return tensor
#' @export
">=.fastai.torch_core.TensorMask" <- function(a, b) {
  a$ge(b)
}

#' @title Greater
#'
#' @param a tensor
#' @param b tensor
#' @return tensor
#' @export
">.fastai.torch_core.TensorMask" <- function(a, b) {
  a$gt(b)
}


#' @title Less or equal
#' @param a tensor
#' @param b tensor
#' @return tensor
#' @export
"<=.fastai.torch_core.TensorMask" <- function(a, b) {
  a$le(b)
}

#' @title Less
#'
#' @param a tensor
#' @param b tensor
#' @return tensor
#' @export
"<.fastai.torch_core.TensorMask" <- function(a, b) {
  a$lt(b)
}


#' @title Max
#'
#' @param a tensor
#' @param ... additional parameters
#' @param na.rm remove NAs
#' @return tensor
#' @export
"max.fastai.torch_core.TensorMask" <- function(a, ..., na.rm = FALSE) {
  a$max()
}

#' @title Min
#'
#' @param a tensor
#' @param ... additional parameters
#' @param na.rm remove NAs
#' @return tensor
#' @export
"min.fastai.torch_core.TensorMask" <- function(a, ..., na.rm = FALSE) {
  a$min()
}

#' @title Dim
#'
#' @param x tensor
#' @return tensor
#' @export
"dim.fastai.torch_core.TensorMask" <- function(x) {
  bt$list(x$shape)
}


#' @title Length
#'
#' @param x tensor
#' @return tensor
#' @export
"length.fastai.torch_core.TensorMask" <- function(x) {
  x$nelement()
}

#' @title Floor divide
#'
#' @param x tensor
#' @param y tensor
#' @return tensor
#' @export
"%/%.fastai.torch_core.TensorMask" <- function(x,y) {
  x$floor_divide(y)
}


#' @title Floor mod
#'
#' @param x tensor
#' @param y tensor
#' @return tensor
#' @export
"%%.fastai.torch_core.TensorMask" <- function(x,y) {
  x$fmod(y)
}

#' @title Logical_and
#'
#' @param x tensor
#' @param y tensor
#' @return tensor
#' @export
"&.fastai.torch_core.TensorMask" <- function(a, b) {
  a$logical_and(b)
}

#' @title Logical_or
#' @param x tensor
#' @param y tensor
#' @return tensor
#' @name or_mask
#' @export
"|.fastai.torch_core.TensorMask" <- function(a, b) {
  a$logical_and(b)
}

#' @title Logical_not
#' @param x tensor
#' @return tensor
#' @name not__mask
#' @export
"!.fastai.torch_core.TensorMask" <- function(a) {
  a$logical_not()
}


#' @title Sort
#'
#' @param x tensor
#' @param decreasing the order
#' @param ... additional parameters to pass
#' @return tensor
#' @export
"sort.fastai.torch_core.TensorMask" <- function(x, decreasing = FALSE, ...) {
  if(decreasing) {
    x$sort(descending = TRUE, ...)
  } else {
    x$sort(...)
  }

}



#' @title Abs
#'
#' @param x tensor
#' @return tensor
#' @export
"abs.fastai.torch_core.TensorMask" <- function(x) {
  x$abs()
}


#' @title Add
#'
#' @param a tensor
#' @param b tensor
#' @return tensor
#' @export
"+.fastai.torch_core.TensorMask" <- function(a, b) {
  a$add(b)
}

#' @title Sub
#' @param a tensor
#' @param b tensor
#' @return tensor
#' @name sub_mask
#' @export
"-.fastai.torch_core.TensorMask" <- function(a, b) {
  a$sub(b)
}

#' @title Div
#'
#' @param a tensor
#' @param b tensor
#' @return tensor
#' @export
"/.fastai.torch_core.TensorMask" <- function(a, b) {
  a$div(b)
}

#' @title Multiply
#'
#' @param a tensor
#' @param b tensor
#' @return tensor
#' @export
"*.fastai.torch_core.TensorMask" <- function(a, b) {
  a$mul(b)
}


#' @title Exp
#'
#' @param x tensor
#' @return tensor
#' @export
"exp.fastai.torch_core.TensorMask" <- function(x) {
  x$exp()
}


#' @title Expm1
#'
#' @param x tensor
#' @return tensor
#' @export
"expm1.fastai.torch_core.TensorMask" <- function(x) {
  x$expm1()
}

#' @title Log
#'
#' @param x tensor
#' @param base base parameter
#' @return tensor
#' @export
"log.fastai.torch_core.TensorMask" <- function(x, base = exp(1)) {
  x$log()
}



#' @title Log1p
#' @param x tensor
#' @return tensor
#' @export
"log1p.fastai.torch_core.TensorMask" <- function(x) {
  x$log1p()
}


#' @title Round
#'
#' @param x tensor
#' @param digits decimal
#' @return tensor
#' @export
"round.fastai.torch_core.TensorMask" <- function(x, digits = 0) {
  x$round()
}


#' @title Sqrt
#'
#' @param x tensor
#' @return tensor
#' @export
"sqrt.fastai.torch_core.TensorMask" <- function(x) {
  x$sqrt()
}


#' @title Floor
#'
#' @param x tensor
#' @return tensor
#' @export
"floor.fastai.torch_core.TensorMask" <- function(x) {
  x$floor()
}

#' @title Ceil
#' @param x tensor
#' @return tensor
#' @export
"ceiling.fastai.torch_core.TensorMask" <- function(x) {
  x$ceil()
}

#' @title Cos
#'
#' @param x tensor
#' @return tensor
#' @export
"cos.fastai.torch_core.TensorMask" <- function(x) {
  x$cos()
}

#' @title Cosh
#' @param x tensor
#' @return tensor
#' @export
"cosh.fastai.torch_core.TensorMask" <- function(x) {
  x$cosh()
}



#' @title Sin
#'
#' @param x tensor
#' @return tensor
#' @export
"sin.fastai.torch_core.TensorMask" <- function(x) {
  x$sin()
}

#' @title Sinh
#'
#' @param x tensor
#' @return tensor
#' @export
"sinh.fastai.torch_core.TensorMask" <- function(x) {
  x$sinh()
}



#' @title Mean of tensor
#'
#' @param x tensor
#' @param ... additional parameters to pass
#' @return tensor
#'
#' @export
"mean.fastai.torch_core.TensorMask" <- function(x, ...) {
  x$mean()
}

