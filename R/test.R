

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


#' @title show_batch
#'
#' @description
#'
#' @param dls dataloader object
#' @param b b
#' @param max_n max_n
#' @param ctxs ctxs
#' @param show show
#' @param unique unique
#'
#' @export
show_batch <- function(dls, b = NULL, max_n = 9, ctxs = NULL,
                       show = TRUE, unique = FALSE, dpi = 90) {

  dls$show_batch(
    b = b,
    max_n = as.integer(max_n),
    ctxs = ctxs,
    show = show,
    unique = unique
  )

  tmp_d = proj_name = gsub(tempdir(), replacement = '/', pattern = '\\', fixed=TRUE)
  fastai2$tabular$all$plt$savefig(paste(tmp_d, 'test.png', sep = '/'), dpi = as.integer(dpi))

  img <- readPNG(paste(tmp_d, 'test.png', sep = '/'))
  lim <- par()
  plot.new()
  rasterImage(img, lim$usr[1], lim$usr[3], lim$usr[2], lim$usr[4])
  invisible(file.remove(paste(tmp_d, 'test.png', sep = '/')))
}






