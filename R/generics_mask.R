
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
"==.fastai.torch_core.TensorMask" <- function(a, b) {
  a$eq(b)
}


#' Pow
#'
#'
#' @export
"^.fastai.torch_core.TensorMask" <- function(a, b) {
  a$pow(b)
}

#' Not equal
#'
#'
#' @name not_equal_to_mask_
#' @export
"!=.fastai.torch_core.TensorMask" <- function(a, b) {
  a$ne(b)
}


#' Greater or equal
#'
#'
#' @export
">=.fastai.torch_core.TensorMask" <- function(a, b) {
  a$ge(b)
}

#' Greater
#'
#'
#' @export
">.fastai.torch_core.TensorMask" <- function(a, b) {
  a$gt(b)
}


#' Less or equal
#'
#' @export
"<=.fastai.torch_core.TensorMask" <- function(a, b) {
  a$le(b)
}

#' Less
#'
#'
#' @export
"<.fastai.torch_core.TensorMask" <- function(a, b) {
  a$lt(b)
}


#' Max
#'
#'
#' @export
"max.fastai.torch_core.TensorMask" <- function(a, ..., na.rm = FALSE) {
  a$max()
}

#' Min
#'
#'
#' @export
"min.fastai.torch_core.TensorMask" <- function(a, ..., na.rm = FALSE) {
  a$min()
}

#' Dim
#'
#'
#' @export
"dim.fastai.torch_core.TensorMask" <- function(x) {
  bt$list(x$shape)
}


#' Length
#'
#'
#' @export
"length.fastai.torch_core.TensorMask" <- function(x) {
  x$nelement()
}

#' Floor divide
#'
#'
#' @export
"%/%.fastai.torch_core.TensorMask" <- function(x,y) {
  x$floor_divide(y)
}


#' Floor mod
#'
#'
#' @export
"%%.fastai.torch_core.TensorMask" <- function(x,y) {
  x$fmod(y)
}

#' Logical_and
#'
#'
#' @export
"&.fastai.torch_core.TensorMask" <- function(a, b) {
  a$logical_and(b)
}

#' Logical_or
#'
#' @name or_mask
#' @export
"|.fastai.torch_core.TensorMask" <- function(a, b) {
  a$logical_and(b)
}

#' Logical_not
#'
#' @name not__mask
#' @export
"!.fastai.torch_core.TensorMask" <- function(a) {
  a$logical_not()
}


#' Sort
#'
#'
#' @export
"sort.fastai.torch_core.TensorMask" <- function(x, decreasing = FALSE, ...) {
  if(decreasing) {
    x$sort(descending = TRUE, ...)
  } else {
    x$sort(...)
  }

}



#' Abs
#'
#'
#' @export
"abs.fastai.torch_core.TensorMask" <- function(a) {
  a$abs()
}


#' Add
#'
#'
#' @export
"+.fastai.torch_core.TensorMask" <- function(a, b) {
  a$add(b)
}

#' Sub
#'
#' @name sub_mask
#' @export
"-.fastai.torch_core.TensorMask" <- function(a, b) {
  a$sub(b)
}

#' Div
#'
#'
#' @export
"/.fastai.torch_core.TensorMask" <- function(a, b) {
  a$div(b)
}

#' Multiply
#'
#'
#' @export
"*.fastai.torch_core.TensorMask" <- function(a, b) {
  a$mul(b)
}


#' Exp
#'
#'
#' @export
"exp.fastai.torch_core.TensorMask" <- function(a) {
  a$exp()
}


#' Expm1
#'
#'
#' @export
"expm1.fastai.torch_core.TensorMask" <- function(a) {
  a$expm1()
}

#' Log
#'
#'
#' @export
"log.fastai.torch_core.TensorMask" <- function(a) {
  a$log()
}



#' Log1p
#'
#'
#' @export
"log1p.fastai.torch_core.TensorMask" <- function(a) {
  a$log1p()
}


#' Round
#'
#'
#' @export
"round.fastai.torch_core.TensorMask" <- function(a) {
  a$round()
}


#' Sqrt
#'
#'
#' @export
"sqrt.fastai.torch_core.TensorMask" <- function(a) {
  a$sqrt()
}


#' Floor
#'
#'
#' @export
"floor.fastai.torch_core.TensorMask" <- function(a) {
  a$floor()
}

#' Ceil
#'
#' @export
"ceiling.fastai.torch_core.TensorMask" <- function(a) {
  a$ceil()
}

#' Cos
#'
#'
#' @export
"cos.fastai.torch_core.TensorMask" <- function(a) {
  a$cos()
}

#' Cosh
#'
#' @export
"cosh.fastai.torch_core.TensorMask" <- function(a) {
  a$cosh()
}



#' Sin
#'
#'
#' @export
"sin.fastai.torch_core.TensorMask" <- function(a) {
  a$sin()
}

#' Sinh
#'
#'
#' @export
"sinh.fastai.torch_core.TensorMask" <- function(a) {
  a$sinh()
}



#' Mean of tensor
#'
#'
#'
#' @export
"mean.fastai.torch_core.TensorMask" <- function(a, ...) {
  a$mean()
}

