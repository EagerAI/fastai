#' @title WandbCallback
#'
#' @description Saves model topology, losses & metrics
#'
#'
#' @param log "gradients" (default), "parameters", "all" or None. Losses & metrics are always logged.
#' @param log_preds whether we want to log prediction samples (default to True).
#' @param log_model whether we want to log our model (default to True). This also requires SaveModelCallback.
#' @param log_dataset Options:
#' - False (default)
#' - True will log folder referenced by learn.dls.path.
#' - a path can be defined explicitly to reference which folder to log.
#' Note: subfolder "models" is always ignored.
#' @param dataset_name name of logged dataset (default to folder name).
#' @param valid_dl DataLoaders containing items used for prediction samples (default to random items from learn.dls.valid.
#' @param n_preds number of logged predictions (default to 36).
#' @param seed used for defining random samples.
#' @param reorder reorder or not
#' @return None
#' @export
WandbCallback <- function(log = "gradients", log_preds = TRUE, log_model = TRUE,
                          log_dataset = FALSE, dataset_name = NULL, valid_dl = NULL,
                          n_preds = 36, seed = 12345, reorder = TRUE) {

  args <- list(
    log = log,
    log_preds = log_preds,
    log_model = log_model,
    log_dataset = log_dataset,
    dataset_name = dataset_name,
    valid_dl = valid_dl,
    n_preds = as.integer(n_preds),
    seed = as.integer(seed),
    reorder = reorder
  )

  do.call(fastai2$callback$wandb$WandbCallback, args)

}

#' @title Wandb login
#'
#' @description Log in to W&B.
#'
#'
#' @param anonymous must,never,allow,false,true
#' @param key API key (secret)
#' @param relogin relogin or not
#' @param host host address
#' @param force whether to force a user to be logged into wandb when running a script
#'
#' @return None
#'
#'
#' @export
login <- function(anonymous = NULL, key = NULL, relogin = NULL, host = NULL, force = NULL) {

 wandb$login(
    anonymous = anonymous,
    key = key,
    relogin = relogin,
    host = host,
    force = force
  )

}


#' @title Wandb init
#'
#' @description Initialize a wandb Run.
#'
#'
#' @param ... parameters to pass
#'
#' @return wandb Run object
#' @section see https://docs.wandb.com/library/init
#' @return None
#' @export
init <- function(...) {

  wandb$init(
    ...
  )

}

#' @title CSVLogger
#'
#' @description Basic class handling tweaks of the training loop by changing a `Learner` in various events
#'
#' @param fname file name
#' @param append append or not
#' @return None
#'
#' @examples
#'
#' \dontrun{
#'
#' URLs_MNIST_SAMPLE()
#' # transformations
#' tfms = aug_transforms(do_flip = FALSE)
#' path = 'mnist_sample'
#' bs = 20
#'
#' #load into memory
#' data = ImageDataLoaders_from_folder(path, batch_tfms = tfms, size = 26, bs = bs)
#'
#'
#' learn = cnn_learner(data, resnet18(), metrics = accuracy, path = getwd())
#'
#' learn %>% fit_one_cycle(2, cbs = CSVLogger())
#'
#' }
#'
#' @export
CSVLogger <- function(fname = "history.csv", append = FALSE) {

  fastai2$callback$all$CSVLogger(
    fname = fname,
    append = append
  )

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

#' @title ReduceLROnPlateau
#'
#' @param ... parameters to pass
#' @return None
#'
#' @examples
#'
#' \dontrun{
#'
#' URLs_MNIST_SAMPLE()
#' # transformations
#' tfms = aug_transforms(do_flip = FALSE)
#' path = 'mnist_sample'
#' bs = 20
#'
#' #load into memory
#' data = ImageDataLoaders_from_folder(path, batch_tfms = tfms, size = 26, bs = bs)
#'
#'
#' learn = cnn_learner(data, resnet18(), metrics = accuracy, path = getwd())
#'
#' learn %>% fit_one_cycle(10, 1e-2, cbs = ReduceLROnPlateau(monitor='valid_loss', patience = 1))
#'
#' }
#'
#' @export
ReduceLROnPlateau <- function(...) {
  fastai2$callback$all$ReduceLROnPlateau(...)
}

#' @title FetchPredsCallback
#'
#' @description A callback to fetch predictions during the training loop
#'
#' @param ds_idx dataset index
#' @param dl DL application
#' @param with_input with input or not
#' @param with_decoded with decoded or not
#' @param cbs callbacks
#' @param reorder reorder or not
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
#' @param with_input include inputs or not
#' @param with_loss include loss or not
#' @param save_preds save predictions
#' @param save_targs save targets/actuals
#' @param concat_dim concatenate dimensions
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







