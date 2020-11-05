

#' @title Timm_learner
#'
#' @description Build a convnet style learner from `dls` and `arch` using the `timm` library
#'
#'
#' @param dls dataloader
#' @param arch model architecture
#' @param ... additional arguments
#' @return None
#' @export
timm_learner <- function(dls, arch, ...) {

  load_pre_models()$timm_learner(
    dls = dls,
    arch = arch,
    ...
  )

}


#' @title Timm models
#'
#' @param ... parameters to pass
#' @return vector
#' @export
timm_list_models <- function(...) {

  args = list(...)

  do.call(timm()$list_models, args)
}

