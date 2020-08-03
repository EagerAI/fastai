

nn <- NULL
fastai<-NULL
tabular<-NULL
vision<-NULL

.onLoad <- function(libname, pkgname) {

  fastai <<- reticulate::import("fastai", delay_load = list(
    priority = 10,
    environment = "r-fastai"
  ))

  nn <<- fastai$train$nn

  # tabular module
  tabular <<- fastai$tabular

  # vision module
  vision <<- fastai$vision

}



