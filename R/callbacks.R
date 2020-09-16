

#' @title CSVLogger
#'
#' @description A `LearnerCallback` that saves history of metrics while training `learn` into CSV `filename`.
#'
#'
#' @param learn learn
#' @param filename filename
#' @param append append
#'
#' @export
CSVLogger <- function(learn, filename = "history", append = FALSE) {

  if(missing(learn)) {
    tabular$callbacks$CSVLogger
  } else {
    args <- list(
      learn = learn,
      filename = filename,
      append = append
    )

    do.call(tabular$callbacks$CSVLogger, args)
  }

}

#' @title CudaCallback
#'
#' @description Move data to CUDA device
#'
#'
#' @param device device
#'
#' @export
CudaCallback <- function(device = NULL) {

  fastai2$callback$all$CudaCallback(
    device = device
  )

}

#' @title HookCallback
#'
#' @description `Callback` that can be used to register hooks on `modules`
#'
#' @details
#'
#' @param modules modules
#' @param every every
#' @param remove_end remove_end
#' @param is_forward is_forward
#' @param detach detach
#' @param cpu cpu
#'
#' @export
HookCallback <- function(modules = NULL, every = NULL, remove_end = TRUE, is_forward = TRUE, detach = TRUE, cpu = TRUE) {

  fastai2$callback$all$HookCallback(
    modules = modules,
    every = every,
    remove_end = remove_end,
    is_forward = is_forward,
    detach = detach,
    cpu = cpu
  )

}

#' @title TrackerCallback
#'
#' @description A `Callback` that keeps track of the best value in `monitor`.
#'
#' @details
#'
#' @param monitor monitor
#' @param comp comp
#' @param min_delta min_delta
#'
#' @export
TrackerCallback <- function(monitor = "valid_loss", comp = NULL, min_delta = 0.0) {

  fastai2$callback$all$TrackerCallback(
    monitor = monitor,
    comp = comp,
    min_delta = min_delta
  )

}


#' @title CollectDataCallback
#'
#'
#'
#' @export
CollectDataCallback <- function(...) {
  fastai2$callback$all$CollectDataCallback(...)
}


#' @title FetchPredsCallback
#'
#' @description A callback to fetch predictions during the training loop
#'
#' @param ds_idx ds_idx
#' @param dl dl
#' @param with_input with_input
#' @param with_decoded with_decoded
#' @param cbs cbs
#' @param reorder reorder
#'
#' @export
FetchPredsCallback <- function(ds_idx = 1, dl = NULL, with_input = FALSE,
                               with_decoded = FALSE, cbs = NULL, reorder = TRUE) {

  python_function_result <- fastai2$callback$all$FetchPredsCallback(
    ds_idx = as.integer(ds_idx),
    dl = dl,
    with_input = with_input,
    with_decoded = with_decoded,
    cbs = cbs,
    reorder = reorder
  )

}


#' @title SaveModelCallback
#'
#'
#'
#' @export
SaveModelCallback <- function(...) {
  fastai2$callback$all$SaveModelCallback(...)
}

#' @title ShowGraphCallback
#'
#'
#'
#' @export
ShowGraphCallback <- function(...) {
  fastai2$callback$all$ShowGraphCallback(...)
}

#' @title TrainEvalCallback
#'
#'
#'
#' @export
TrainEvalCallback <- function(...) {
  fastai2$callback$all$TrainEvalCallback(...)
}

#' @title ShortEpochCallback
#'
#'
#'
#' @export
ShortEpochCallback <- function(...) {
  fastai2$callback$all$ShortEpochCallback(...)
}



#' @title GatherPredsCallback
#'
#' @description `Callback` that saves the predictions and targets, optionally `with_loss`
#'
#' @details
#'
#' @param with_input with_input
#' @param with_loss with_loss
#' @param save_preds save_preds
#' @param save_targs save_targs
#' @param concat_dim concat_dim
#'
#' @export
GatherPredsCallback <- function(with_input = FALSE, with_loss = FALSE,
                                save_preds = NULL, save_targs = NULL, concat_dim = 0) {

  fastai2$callback$all$GatherPredsCallback(
    with_input = with_input,
    with_loss = with_loss,
    save_preds = save_preds,
    save_targs = save_targs,
    concat_dim = as.integer(concat_dim)
  )

}

#' @title EarlyStoppingCallback
#'
#'
#'
#' @export
EarlyStoppingCallback <- function(...) {
  fastai2$callback$all$EarlyStoppingCallback(...)
}

#' @title TerminateOnNaNCallback
#'
#'
#'
#' @export
TerminateOnNaNCallback <- function(...) {
  fastai2$callback$all$TerminateOnNaNCallback(...)
}







