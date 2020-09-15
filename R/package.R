

nn <- NULL
fastai2<-NULL
tabular<-NULL
vision<-NULL
text<-NULL
Module<-NULL
medical<-NULL

.onLoad <- function(libname, pkgname) {

  fastai2 <<- reticulate::import("fastai", delay_load = list(
    priority = 10,
    environment = "r-fastai"
  ))

  if(reticulate::py_module_available('IPython') &
     reticulate::py_module_available('torch') &
     reticulate::py_module_available('torchvision') &
     reticulate::py_module_available('fastai')) {
    # tabular module
    tabular <<- fastai2$tabular$all

    # vision module
    vision <<- fastai2$vision

    # collab module
    collab <<- fastai2$collab

    # text module
    text <<- fastai2$text$all

    # Torch module
    nn <<- fastai2$torch_core$nn

    # Metrics
    metrics <<- fastai2$metrics

    # Module
    Module <<- fastai2$vision$all$Module

    # Medical
    medical <<- fastai2$medical$imaging

    # windows
    dicom_windows <<- fastai2$medical$imaging$dicom_windows

    # cmap
    cm <<- fastai2$vision$all$plt$cm

    # colors
    colors <<- fastai2$vision$all$matplotlib$colors

    # remove fill
    fastaip <<- reticulate::import('fastprogress')

    fastaip$progress_bar$fill = ''
  }

}



