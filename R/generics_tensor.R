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


#' Pow
#'
#'
#' @name pow
#' @export
"^.torch.Tensor" <- function(a, b) {
  a$pow(b)
}

#' Not equal
#'
#'
#' @name not_equal_to
#' @export
"!=.torch.Tensor" <- function(a, b) {
  a$ne(b)
}


#' Greater or equal
#'
#'
#' @name greater_or_equal
#' @export
">=.torch.Tensor" <- function(a, b) {
  a$ge(b)
}

#' Greater
#'
#'
#' @name greater
#' @export
">.torch.Tensor" <- function(a, b) {
  a$gt(b)
}


#' Less or equal
#'
#'
#' @name less_or_equal
#' @export
"<=.torch.Tensor" <- function(a, b) {
  a$le(b)
}

#' Less
#'
#'
#' @name greater
#' @export
"<.torch.Tensor" <- function(a, b) {
  a$lt(b)
}


#' Max
#'
#'
#' @name max
#' @export
"max.torch.Tensor" <- function(a, ..., na.rm = FALSE) {
  a$max()
}

#' Min
#'
#'
#' @name min
#' @export
"min.torch.Tensor" <- function(a, ..., na.rm = FALSE) {
  a$min()
}

#' Dim
#'
#'
#' @name dim
#' @export
"dim.torch.Tensor" <- function(x) {
  bt$list(x$shape)
}



#' Length
#'
#'
#' @name length
#' @export
"length.torch.Tensor" <- function(x) {
  x$nelement()
}

#' Floor divide
#'
#'
#' @name floor_div
#' @export
"%/%.torch.Tensor" <- function(x,y) {
  x$floor_divide(y)
}


#' Floor mod
#'
#'
#' @name floor_mod
#' @export
"%%.torch.Tensor" <- function(x,y) {
  x$fmod(y)
}

#' Logical_and
#'
#'
#' @name logical_and
#' @export
"&.torch.Tensor" <- function(a, b) {
  a$logical_and(b)
}

#' Logical_or
#'
#'
#' @name logical_or
#' @export
"|.torch.Tensor" <- function(a, b) {
  a$logical_and(b)
}

#' Logical_not
#'
#'
#' @name logical_or
#' @export
"!.torch.Tensor" <- function(a) {
  a$logical_not()
}



#' Sort
#'
#'
#' @name sort
#' @export
"sort.torch.Tensor" <- function(x, decreasing = FALSE, ...) {
  x$sort(...)
}



#' Abs
#'
#'
#' @name abs
#' @export
"abs.torch.Tensor" <- function(a) {
  a$abs()
}


#' Add
#'
#'
#' @name add
#' @export
"+.torch.Tensor" <- function(a, b) {
  a$add(b)
}

#' Sub
#'
#'
#' @name sub
#' @export
"-.torch.Tensor" <- function(a, b) {
  a$sub(b)
}

#' Div
#'
#'
#' @name div
#' @export
"/.torch.Tensor" <- function(a, b) {
  a$div(b)
}

#' Multiply
#'
#'
#' @name div
#' @export
"*.torch.Tensor" <- function(a, b) {
  a$mul(b)
}


#' Exp
#'
#'
#' @name exp
#' @export
"exp.torch.Tensor" <- function(a, b) {
  a$exp(b)
}


#' Expm1
#'
#'
#' @name expm1
#' @export
"expm1.torch.Tensor" <- function(a, b) {
  a$expm1(b)
}

#' Log
#'
#'
#' @name log
#' @export
"log.torch.Tensor" <- function(a, b) {
  a$log(b)
}

#' Log10
#'
#'
#' @name log10
#' @export
"log10.torch.Tensor" <- function(a, b) {
  a$log10(b)
}

#' Log1p
#'
#'
#' @name log1p
#' @export
"log1p.torch.Tensor" <- function(a, b) {
  a$log1p(b)
}

#' Log2
#'
#'
#' @name log2
#' @export
"log2.torch.Tensor" <- function(a, b) {
  a$log2(b)
}

#' Round
#'
#'
#' @name round
#' @export
"round.torch.Tensor" <- function(a) {
  a$round()
}


#' Sqrt
#'
#'
#' @name sqrd
#' @export
"sqrt.torch.Tensor" <- function(a) {
  a$sqrt()
}


#' Floor
#'
#'
#' @name add
#' @export
"floor.torch.Tensor" <- function(a) {
  a$floor()
}

#' Ceil
#'
#'
#' @name add
#' @export
"ceiling.torch.Tensor" <- function(a) {
  a$ceil()
}

#' Cos
#'
#'
#' @name add
#' @export
"cos.torch.Tensor" <- function(a) {
  a$cos()
}

#' Cosh
#'
#'
#' @name add
#' @export
"cosh.torch.Tensor" <- function(a) {
  a$cosh()
}



#' Sin
#'
#'
#' @name add
#' @export
"sin.torch.Tensor" <- function(a) {
  a$sin()
}

#' Sinh
#'
#'
#' @name add
#' @export
"sinh.torch.Tensor" <- function(a) {
  a$sinh()
}



#' Mean of tensor
#'
#'
#'
#' @export
"mean.torch.Tensor" <- function(a) {
  a$mean()
}


#' Median of tensor
#'
#'
#'
#' @export
"median.torch.Tensor" <- function(a) {
  a$median()
}

#' Mode of tensor
#'
#'
#'
#' @export
"mode.torch.Tensor" <- function(a) {
  a$mode()
}


#' Std of tensor
#'
#'
#'
#' @export
"std.torch.Tensor" <- function(a) {
  a$std()
}

#' Unique of tensor
#'
#'
#'
#' @export
"unique.torch.Tensor" <- function(a) {
  a$unique()
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
to_matrix = function(obj, matrix = TRUE) {
  if(inherits(obj,'pydicom.dataset.FileDataset')) {
    res = obj$as_dict()

    get_names = names(res)

    res = lapply(1:length(res), function(x) ifelse(inherits(res[[x]], "python.builtin.object"),
                                                   as.character(res[[x]]), res[[x]]))
    names(res) = get_names

    res = as.data.frame(do.call(cbind, res))

    tmp = gsub(tempdir(), replacement = '/',pattern = '\\', fixed = TRUE)

    write.csv(res, paste(tmp,'temp.csv',sep = '/'), row.names = FALSE)

    res = read.csv(paste(tmp,'temp.csv',sep = '/'))

    if(matrix) {
      as.matrix(res)
    } else {
      res
    }

  } else {
    if(matrix) {
      fastai2$basics$tensor(obj)$cpu()$numpy()
    } else {
      as.data.frame(fastai2$basics$tensor(obj)$cpu()$numpy())
    }
  }
}

#' @title Print model
#'
#'
#' @export
print.fastai.learner.Learner = function(x, ...) {
  print(x$model)
}


