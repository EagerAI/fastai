
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
#' @name pow
#' @export
"^.fastai.torch_core.TensorMask" <- function(a, b) {
  a$pow(b)
}

#' Not equal
#'
#'
#' @name not_equal_to
#' @export
"!=.fastai.torch_core.TensorMask" <- function(a, b) {
  a$ne(b)
}


#' Greater or equal
#'
#'
#' @name greater_or_equal
#' @export
">=.fastai.torch_core.TensorMask" <- function(a, b) {
  a$ge(b)
}

#' Greater
#'
#'
#' @name greater
#' @export
">.fastai.torch_core.TensorMask" <- function(a, b) {
  a$gt(b)
}


#' Less or equal
#'
#'
#' @name less_or_equal
#' @export
"<=.fastai.torch_core.TensorMask" <- function(a, b) {
  a$le(b)
}

#' Less
#'
#'
#' @name greater
#' @export
"<.fastai.torch_core.TensorMask" <- function(a, b) {
  a$lt(b)
}


#' Max
#'
#'
#' @name max
#' @export
"max.fastai.torch_core.TensorMask" <- function(a, ..., na.rm = FALSE) {
  a$max()
}

#' Min
#'
#'
#' @name min
#' @export
"min.fastai.torch_core.TensorMask" <- function(a, ..., na.rm = FALSE) {
  a$min()
}

#' Dim
#'
#'
#' @name dim
#' @export
"dim.fastai.torch_core.TensorMask" <- function(x) {
  x$dim()
}

#' Length
#'
#'
#' @name length
#' @export
"length.fastai.torch_core.TensorMask" <- function(x) {
  x$nelement()
}

#' Floor divide
#'
#'
#' @name floor_div
#' @export
"%/%.fastai.torch_core.TensorMask" <- function(x,y) {
  x$floor_divide(y)
}


#' Floor mod
#'
#'
#' @name floor_mod
#' @export
"%%.fastai.torch_core.TensorMask" <- function(x,y) {
  x$fmod(y)
}

#' Logical_and
#'
#'
#' @name logical_and
#' @export
"&.fastai.torch_core.TensorMask" <- function(a, b) {
  a$logical_and(b)
}

#' Logical_or
#'
#'
#' @name logical_or
#' @export
"|.fastai.torch_core.TensorMask" <- function(a, b) {
  a$logical_and(b)
}

#' Logical_not
#'
#'
#' @name logical_or
#' @export
"!.fastai.torch_core.TensorMask" <- function(a) {
  a$logical_not()
}


#' Matmul
#'
#'
#' @name matmul
#' @export
"%*%.fastai.torch_core.TensorMask" <- function(a, b) {
  a$matmul(b)
}


#' Sort
#'
#'
#' @name sort
#' @export
"sort.fastai.torch_core.TensorMask" <- function(x, decreasing = FALSE, ...) {
  x$sort(...)
}



#' Abs
#'
#'
#' @name abs
#' @export
"abs.fastai.torch_core.TensorMask" <- function(a) {
  a$abs()
}


#' Add
#'
#'
#' @name add
#' @export
"+.fastai.torch_core.TensorMask" <- function(a, b) {
  a$add(b)
}

#' Sub
#'
#'
#' @name sub
#' @export
"-.fastai.torch_core.TensorMask" <- function(a, b) {
  a$sub(b)
}

#' Div
#'
#'
#' @name div
#' @export
"/.fastai.torch_core.TensorMask" <- function(a, b) {
  a$div(b)
}

#' Multiply
#'
#'
#' @name div
#' @export
"*.fastai.torch_core.TensorMask" <- function(a, b) {
  a$mul(b)
}


#' Exp
#'
#'
#' @name exp
#' @export
"exp.fastai.torch_core.TensorMask" <- function(a, b) {
  a$exp(b)
}


#' Expm1
#'
#'
#' @name expm1
#' @export
"expm1.fastai.torch_core.TensorMask" <- function(a, b) {
  a$expm1(b)
}

#' Log
#'
#'
#' @name log
#' @export
"log.fastai.torch_core.TensorMask" <- function(a, b) {
  a$log(b)
}

#' Log10
#'
#'
#' @name log10
#' @export
"log10.fastai.torch_core.TensorMask" <- function(a, b) {
  a$log10(b)
}

#' Log1p
#'
#'
#' @name log1p
#' @export
"log1p.fastai.torch_core.TensorMask" <- function(a, b) {
  a$log1p(b)
}

#' Log2
#'
#'
#' @name log2
#' @export
"log2.fastai.torch_core.TensorMask" <- function(a, b) {
  a$log2(b)
}

#' Round
#'
#'
#' @name round
#' @export
"round.fastai.torch_core.TensorMask" <- function(a) {
  a$round()
}


#' Sqrt
#'
#'
#' @name sqrd
#' @export
"sqrt.fastai.torch_core.TensorMask" <- function(a) {
  a$sqrt()
}


#' Floor
#'
#'
#' @name add
#' @export
"floor.fastai.torch_core.TensorMask" <- function(a) {
  a$floor()
}

#' Ceil
#'
#'
#' @name add
#' @export
"ceiling.fastai.torch_core.TensorMask" <- function(a) {
  a$ceil()
}

#' Cos
#'
#'
#' @name add
#' @export
"cos.fastai.torch_core.TensorMask" <- function(a) {
  a$cos()
}

#' Cosh
#'
#'
#' @name add
#' @export
"cosh.fastai.torch_core.TensorMask" <- function(a) {
  a$cosh()
}



#' Sin
#'
#'
#' @name add
#' @export
"sin.fastai.torch_core.TensorMask" <- function(a) {
  a$sin()
}

#' Sinh
#'
#'
#' @name add
#' @export
"sinh.fastai.torch_core.TensorMask" <- function(a) {
  a$sinh()
}



#' Mean of tensor
#'
#'
#'
#' @export
"mean.fastai.torch_core.TensorMask" <- function(a) {
  a$mean()
}


#' Median of tensor
#'
#'
#'
#' @export
"median.fastai.torch_core.TensorMask" <- function(a) {
  a$median()
}

#' Mode of tensor
#'
#'
#'
#' @export
"mode.fastai.torch_core.TensorMask" <- function(a) {
  a$mode()
}


#' Std of tensor
#'
#'
#'
#' @export
"std.fastai.torch_core.TensorMask" <- function(a) {
  a$std()
}

#' Unique of tensor
#'
#'
#'
#' @export
"unique.fastai.torch_core.TensorMask" <- function(a) {
  a$unique()
}


