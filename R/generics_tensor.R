#' @title Equal
#'
#' @param a tensor
#' @param b tensor
#' @return tensor
#' @export
"==.torch.Tensor" <- function(a, b) {
  a$eq(b)
}


#' @title Pow
#'
#' @param a tensor
#' @param b tensor
#' @return tensor
#' @name pow
#' @export
"^.torch.Tensor" <- function(a, b) {
  a$pow(b)
}

#' @title Not equal
#'
#' @param a tensor
#' @param b tensor
#' @return tensor
#' @name not_equal_to
#' @export
"!=.torch.Tensor" <- function(a, b) {
  a$ne(b)
}


#' @title Greater or equal
#'
#' @param a tensor
#' @param b tensor
#' @return tensor
#' @name greater_or_equal
#' @export
">=.torch.Tensor" <- function(a, b) {
  a$ge(b)
}

#' @title Greater
#'
#' @param a tensor
#' @param b tensor
#' @return tensor
#' @name greater
#' @export
">.torch.Tensor" <- function(a, b) {
  a$gt(b)
}


#' @title Less or equal
#' @param a tensor
#' @param b tensor
#' @return tensor
#' @name less_or_equal
#' @export
"<=.torch.Tensor" <- function(a, b) {
  a$le(b)
}

#' @title Less
#'
#' @param a tensor
#' @param b tensor
#' @return tensor
#' @name less
#' @export
"<.torch.Tensor" <- function(a, b) {
  a$lt(b)
}


#' @title Max
#'
#' @param a tensor
#' @param ... additional parameters
#' @param na.rm remove NAs
#' @return tensor
#' @name max
#' @export
"max.torch.Tensor" <- function(a, ..., na.rm = FALSE) {
  a$max()
}

#' @title Min
#'
#' @param a tensor
#' @param ... additional parameters
#' @param na.rm remove NAs
#' @return tensor
#' @name min
#' @export
"min.torch.Tensor" <- function(a, ..., na.rm = FALSE) {
  a$min()
}

#' @title Dim
#'
#' @param x tensor
#' @return tensor
#'
#' @name dim
#' @export
"dim.torch.Tensor" <- function(x) {
  bt$list(x$shape)
}




#' @title Length
#'
#' @param x tensor
#' @return tensor
#'
#' @name length
#' @export
"length.torch.Tensor" <- function(x) {
  x$nelement()
}

#' @title Floor divide
#'
#' @param x tensor
#' @param y tensor
#' @return tensor
#' @name floor_div
#' @export
"%/%.torch.Tensor" <- function(x, y) {
  x$floor_divide(y)
}


#' @title Floor mod
#'
#' @param x tensor
#' @param y tensor
#' @return tensor
#' @name floor_mod
#' @export
"%%.torch.Tensor" <- function(x, y) {
  x$fmod(y)
}

#' @title Logical_and
#'
#' @param x tensor
#' @param y tensor
#' @return tensor
#'
#' @name logical_and
#' @export
"&.torch.Tensor" <- function(x, y) {
  x$logical_and(y)
}

#' @title Logical_or
#' @param x tensor
#' @param y tensor
#' @return tensor
#' @name logical_or
#' @export
"|.torch.Tensor" <- function(x, y) {
  x$logical_or(y)
}

#' @title Logical_not
#' @param x tensor
#' @return tensor
#' @name logical_not_
#' @export
"!.torch.Tensor" <- function(x) {
  x$logical_not()
}



#' @title Sort
#'
#' @param x tensor
#' @param decreasing the order
#' @param ... additional parameters to pass
#' @name sort
#' @export
"sort.torch.Tensor" <- function(x, decreasing = FALSE, ...) {
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
#' @name abs
#' @export
"abs.torch.Tensor" <- function(x) {
  x$abs()
}


#' @title Add
#'
#' @param a tensor
#' @param b tensor
#' @return tensor
#' @name add
#' @export
"+.torch.Tensor" <- function(a, b) {
  a$add(b)
}

#' @title Sub
#' @param a tensor
#' @param b tensor
#' @return tensor
#' @name sub
#' @export
"-.torch.Tensor" <- function(a, b) {
  a$sub(b)
}

#' @title Div
#'
#' @param a tensor
#' @param b tensor
#' @return tensor
#' @name div
#' @export
"/.torch.Tensor" <- function(a, b) {
  a$div(b)
}

#' @title Multiply
#'
#' @param a tensor
#' @param b tensor
#' @return tensor
#' @name multiplygit add -A && git commit -m 'staging all files'
#' @export
"*.torch.Tensor" <- function(a, b) {
  a$mul(b)
}


#' @title Exp
#'
#' @param x tensor
#' @return tensor
#' @name exp
#' @export
"exp.torch.Tensor" <- function(x) {
  x$exp()
}


#' @title Expm1
#'
#' @param x tensor
#' @return tensor
#' @name expm1
#' @export
"expm1.torch.Tensor" <- function(x) {
  x$expm1()
}

#' @title Log
#'
#' @param x tensor
#' @param base base parameter
#' @return tensor
#' @name log
#' @export
"log.torch.Tensor" <- function(x, base = exp(1)) {
  x$log()
}


#' @title Log1p
#' @param x tensor
#' @return tensor
#' @name log1p
#' @export
"log1p.torch.Tensor" <- function(x) {
  x$log1p()
}


#' @title Round
#'
#' @param x tensor
#' @param digits decimal
#' @return tensor
#' @name round
#' @export
"round.torch.Tensor" <- function(x, digits = 0) {
  x$round()
}


#' @title Sqrt
#'
#' @param x tensor
#' @return tensor
#' @name sqrd
#' @export
"sqrt.torch.Tensor" <- function(x) {
  x$sqrt()
}


#' @title Floor
#'
#' @param x tensor
#' @return tensor
#' @name floor_
#' @export
"floor.torch.Tensor" <- function(x) {
  x$floor()
}

#' @title Ceil
#' @param x tensor
#' @return tensor
#' @name ceiling_
#' @export
"ceiling.torch.Tensor" <- function(x) {
  x$ceil()
}

#' @title Cos
#'
#' @param x tensor
#' @return tensor
#' @name cos_
#' @export
"cos.torch.Tensor" <- function(x) {
  x$cos()
}

#' @title Cosh
#' @param x tensor
#' @return tensor
#' @name cosh_
#' @export
"cosh.torch.Tensor" <- function(x) {
  x$cosh()
}



#' @title Sin
#'
#' @param x tensor
#' @return tensor
#' @name sin_
#' @export
"sin.torch.Tensor" <- function(x) {
  x$sin()
}

#' @title Sinh
#'
#' @param x tensor
#' @return tensor
#' @name add
#' @export
"sinh.torch.Tensor" <- function(x) {
  x$sinh()
}



#' @title Mean of tensor
#'
#' @param x tensor
#' @param ... additional parameters to pass
#' @return tensor
#'
#'
#' @export
"mean.torch.Tensor" <- function(x, ...) {
  x$mean()
}





#' @title Tensor to float
#'
#' @param tensor tensor
#' @return tensor
#' @export
float <- function(tensor) {
  tensor$float()
}



#' @title To matrix
#' @param obj learner/model
#' @param matrix bool, to R matrix
#' @importFrom  utils write.csv read.csv
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
#' @param x object
#' @param ... additional parameters to pass
#' @return None
#' @export
print.fastai.learner.Learner = function(x, ...) {
  res = try(x$model(),  silent = TRUE)
  if(inherits(res,'try-error')) {
    print(x$model)
  } else {
    print(x$model())
  }
}


