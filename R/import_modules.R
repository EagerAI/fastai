

#' @title Python path
#'
#'
#' @return None
python_path <- function() {
  system.file("python", package = "fastai")
}

#' @title NN module
#'
#'
#'
#' @return None
Module_test <- function() try(reticulate::import_from_path('fastaibuilt', path = python_path())$Module_test, TRUE)

#' @title Bs finder
#'
#'
#'
#' @return None
bs_finder <- function() try(reticulate::import_from_path('fastaibuilt', path = python_path())$bs_finder, TRUE)



#' @title Timm models
#'
#'
#' @return None
load_pre_models <- function() try(reticulate::import_from_path('fastaibuilt',
                                                               path = python_path())$pretrained_timm_models, TRUE)




#' @title Timm module
#'
#' @return None
timm <- function() try(reticulate::import('timm'), TRUE)


#' @title Transformer module
#'
#' @return None
hug <- function() try(reticulate::import_from_path('fastaibuilt', path = python_path())$Transformer, TRUE)

#' @title Crappify module
#'
#' @return None
crap <- function() try(reticulate::import_from_path('fastaibuilt', path = python_path())$crappify, TRUE)



#' @title Ignite module
#'
#' @return None
migrating_ignite <- function() try(reticulate::import_from_path('fastaibuilt', path = python_path())$migrating_ignite, TRUE)

#' @title Lightning module
#'
#' @return None
migrating_lightning <- function() try(reticulate::import_from_path('fastaibuilt', path = python_path())$migrating_lightning, TRUE)

#' @title Pytorch module
#'
#' @return None
migrating_pytorch <- function() try(reticulate::import_from_path('fastaibuilt', path = python_path())$migrating_pytorch, TRUE)

#' @title Catalyst module
#'
#' @return None
catalyst <- function() try(reticulate::import_from_path('fastaibuilt', path = python_path())$migrating_catalyst, TRUE)

#' @title Retinanet module
#'
#' @return None
retinanet_ <- function() try(reticulate::import_from_path('fastaibuilt', path = python_path())$retinanet, TRUE)


#' @title Wandb module
#'
#' @return None
wandb <- function() try(reticulate::import('wandb'), TRUE)



#' @title Wandb module
#'
#' @return None
fastinf <- function() try(reticulate::import('fastinference'), TRUE)


#' @title Shap module
#'
#' @return None
shap <- function() try(reticulate::import('shap'), TRUE)


#' @title Fastaudio module
#'
#' @return None
fastaudio <- function() try(reticulate::import('fastaudio'), TRUE)

#' @title Kaggle module
#' @export
#' @return None
kg <- function() try(reticulate::import('kaggle'), TRUE)


#' @title Upit module
#'
#' @return None
upit <- function() try(reticulate::import('upit'), TRUE)

#' @title Timeseries module
#'
#' @return None
tms <- function() try(reticulate::import('timeseries_fastai'), TRUE)


#' @title Blurr module
#'
#' @return None
blurr <- function() try(reticulate::import('blurr'), TRUE)

#' @title Builtins module
#'
#' @return None
bt <- function() try(reticulate::import_builtins(), TRUE)

#' @title Slice
#'
#' @param ... additional arguments
#' @details slice(start, stop[, step]) Create a slice object. This is used for extended slicing (e.g. a[0:10:2]).
#'
#' @return sliced object
#' @export
slice <- function(...) {

  args = list(...)

  do.call(bt()$slice, args)

}



#' @title TransformersTokenizer
#'
#'
#' @param tokenizer tokenizer object
#' @return None
#' @export
TransformersTokenizer <- function(tokenizer) {

  hug()$TransformersTokenizer(
    tokenizer = tokenizer
  )

}

#' @title TransformersDropOutput
#'
#'
#' @return None
#' @export
TransformersDropOutput <- function() {

  hug()$TransformersDropOutput()

}
########################################### MAIN MODULES #########################################################

#' @title Builtins module
#' @export
#' @return None
torch <- function() {
  try(torch <- reticulate::import('fastai.torch_basics'), TRUE)
  try(torch <- torch$torch, TRUE)
}

#' @title Tabular
#'
#' @return None
tabular <- function() {
  try(tabular <- reticulate::import('fastai.tabular'), TRUE)
  try(tabular <- tabular$all, TRUE)
}

#' @title Vision module
#'
#' @return None
vision <- function() {
  try(vision <- reticulate::import('fastai.vision'), TRUE)
}

#' @title Collab module
#'
#' @return None
collab <- function() {
  try(collab <- reticulate::import('fastai.collab'), TRUE)
}

#' @title Text module
#'
#' @return None
text <- function() {
  try(text <- reticulate::import('fastai.text'), TRUE)
  try(text <- text$all, TRUE)
}

#' @title NN module
#' @export
#' @return None
nn <- function() {
  try(nn <- reticulate::import('fastai.torch_core'), TRUE)
  try(nn <- nn$nn, TRUE)
}

#' @title Metrics module
#'
#' @return None
metrics <- function() {
  try(metrics <- reticulate::import('fastai.metrics'), TRUE)
}

#' @title Module module
#' @export
#' @return None
Module <- function() {
  try(Module <- reticulate::import('fastai.vision'), TRUE)
  try(Module <- Module$all$Module, TRUE)
}

#' @title Medical module
#'
#' @return None
medical <- function() {
  try(medical <- reticulate::import('fastai.medical'), TRUE)
  try(medical <- medical$imaging, TRUE)
}

#' @title Dicom_windows module
#' @export
#' @return None
dicom_windows <- function() {
  try(medical <- reticulate::import('fastai.medical'), TRUE)
  try(medical <- medical$imaging$dicom_windows, TRUE)
}


#' @title Cm module
#' @export
#' @return None
cm <- function() {
  try(vision <- reticulate::import('fastai.vision'), TRUE)
  try(vision <- vision$all$plt$cm, TRUE)
}

#' @title Colors module
#' @export
#' @return None
colors <- function() {
  try(vision <- reticulate::import('fastai.vision'), TRUE)
  try(vision <- vision$all$matplotlib$colors, TRUE)
}


#' @title Callback module
#' @export
#' @return None
Callback <- function() {
  try(Callback <- reticulate::import('fastai.callback'), TRUE)
  try(Callback <- Callback$all$Callback, TRUE)
}




#' @title RemoveType module
#' @export
#' @return None
RemoveType <- function() {
  try(augment <- reticulate::import('fastaudio.augment'), TRUE)
  try(augment <- augment$preprocess$RemoveType, TRUE)
}

#' @title AudioPadType module
#' @export
#' @return None
AudioPadType <- function() {
  try(augment <- reticulate::import('fastaudio.augment'), TRUE)
  try(augment <- augment$signal$AudioPadType, TRUE)
}

#' @title NoiseColor module
#' @export
#' @return None
NoiseColor <- function() {
  try(augment <- reticulate::import('fastaudio.augment'), TRUE)
  try(augment <- augment$signal$NoiseColor, TRUE)
}

#' @title AudioSpectrogram module
#'
#' @return None
AudioSpectrogram <- function() {
  try(core <- reticulate::import('fastaudio.core'), TRUE)
  try(core <- core$spectrogram$AudioSpectrogram, TRUE)
}



