

nn <- NULL
fastai<-NULL
tabular<-NULL
vision<-NULL

.onLoad <- function(libname, pkgname) {

  fastai2 <<- reticulate::import("fastai2", delay_load = list(
    priority = 10,
    environment = "r-fastai"
  ))

  nn <<- fastai2$torch_core$nn

  # tabular module
  tabular <<- fastai2$tabular$all

  # vision module
  vision <<- fastai2$vision

}



