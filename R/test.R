

#' @export
"+.torch.nn.modules.container.Sequential" <- function(a, b) {

  res = a$`__dict__`$`_modules`

  if(length(names(res))>0) {
    ll = names(res)
    ll = suppressWarnings(as.numeric(ll))
    ll = ll[!is.na(ll)]
    if(length(ll) > 0) {
      max_ = as.character(max(ll) + 1)
    } else {
      max_ = '0'
    }

  } else {
    max_ = '0'
  }

  if(is.list(b)) {
    max_ = b[[1]]
    b = b[[2]]
  }

  a$add_module(max_, module = b)
  a
}



#' @title Get history
#'
#'
#' @export
to_fastai_training_history <- function(history) {
  structure(class = "fastai_training_history", list(
    history = history
  ))
}


#' @title Plot history
#'
#'
#' @export
plot.to_fastai_training_history <- function(history) {
  plot.ts(history)
}




