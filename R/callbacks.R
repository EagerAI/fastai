

#' @title CSVLogger
#'
#' @description A `LearnerCallback` that saves history of metrics while training `learn` into CSV `filename`.
#'
#'
#' @param learn learner object
#' @param filename the name of the file "history.csv"
#' @param append whether to append the new file or not
#' @return None
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
#' @param device device name
#' @return None
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
#'
#' @param modules the modules
#' @param every int, every epoch
#' @param remove_end logical, remove_end
#' @param is_forward logical, is_forward
#' @param detach detach
#' @param cpu to cpu or not
#' @return None
#' @export
HookCallback <- function(modules = NULL, every = NULL, remove_end = TRUE,
                         is_forward = TRUE, detach = TRUE, cpu = TRUE) {

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
#'
#' @param monitor monitor the loss
#' @param comp comp
#' @param min_delta minimum delta
#' @return None
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
#' @param ... parameters to pass
#' @return None
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
#' @return None
#' @export
FetchPredsCallback <- function(ds_idx = 1, dl = NULL, with_input = FALSE,
                               with_decoded = FALSE, cbs = NULL, reorder = TRUE) {

  fastai2$callback$all$FetchPredsCallback(
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
#' @param ... parameters to pass
#' @return None
#' @export
SaveModelCallback <- function(...) {
  fastai2$callback$all$SaveModelCallback(...)
}

#' @title ShowGraphCallback
#'
#'
#' @param ... parameters to pass
#' @return None
#' @export
ShowGraphCallback <- function(...) {
  fastai2$callback$all$ShowGraphCallback(...)
}

#' @title TrainEvalCallback
#' @param ... parameters to pass
#' @return None
#'
#'
#' @export
TrainEvalCallback <- function(...) {
  fastai2$callback$all$TrainEvalCallback(...)
}

#' @title ShortEpochCallback
#'
#' @param ... parameters to pass
#' @return None
#' @export
ShortEpochCallback <- function(...) {
  fastai2$callback$all$ShortEpochCallback(...)
}



#' @title GatherPredsCallback
#'
#' @description `Callback` that saves the predictions and targets, optionally `with_loss`
#'
#' @param with_input with_input
#' @param with_loss with_loss
#' @param save_preds save_preds
#' @param save_targs save_targs
#' @param concat_dim concat_dim
#' @return None
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
#' @param ... parameters to pass
#' @return None
#' @export
EarlyStoppingCallback <- function(...) {
  fastai2$callback$all$EarlyStoppingCallback(...)
}

#' @title TerminateOnNaNCallback
#'
#' @param ... parameters to pass
#' @return None
#'
#' @export
TerminateOnNaNCallback <- function(...) {
  fastai2$callback$all$TerminateOnNaNCallback(...)
}







