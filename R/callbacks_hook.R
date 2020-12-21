

#' @title Hook
#'
#' @description Create a hook on `m` with `hook_func`.
#' @details Hooks are functions you can attach to a particular layer in your
#' model and that will be executed in the forward pass (for forward hooks)
#' or backward pass (for backward hooks).
#'
#' @param m m aprameter
#' @param hook_func hook function
#' @param is_forward is_forward or not
#' @param detach detach or not
#' @param cpu cpu or not
#' @param gather gather or not
#' @return None
#' @export
Hook <- function(m, hook_func, is_forward = TRUE, detach = TRUE, cpu = FALSE, gather = FALSE) {

  args <- list(
    m = m,
    hook_func = hook_func,
    is_forward = is_forward,
    detach = detach,
    cpu = cpu,
    gather = gather
  )

  do.call(fastai2$callback$hook$Hook, args)

}


#' @title HookCallback
#'
#' @description `Callback` that can be used to register hooks on `modules`
#'
#'
#' @param modules modules
#' @param every every
#' @param remove_end remove_end or not
#' @param is_forward is_forward or not
#' @param detach detach or not
#' @param cpu cpu or not
#' @return None
#' @export
HookCallback <- function(modules = NULL, every = NULL, remove_end = TRUE,
                         is_forward = TRUE, detach = TRUE, cpu = TRUE) {

  args <- list(
    modules = modules,
    every = every,
    remove_end = remove_end,
    is_forward = is_forward,
    detach = detach,
    cpu = cpu
  )

  if(is.null(args$modules))
    fastai2$callback$hook$HookCallback
  else
    do.call(fastai2$callback$hook$HookCallback,args)

}


#' @title Hooks
#'
#' @description Create several hooks on the modules in `ms` with `hook_func`.
#'
#'
#' @param ms ms parameter
#' @param hook_func hook function
#' @param is_forward is_forward or not
#' @param detach detach or not
#' @param cpu cpu or not
#' @return None
#' @export
Hooks <- function(ms, hook_func, is_forward = TRUE, detach = TRUE, cpu = FALSE) {

  args <-list(
    ms = ms,
    hook_func = hook_func,
    is_forward = is_forward,
    detach = detach,
    cpu = cpu
  )

  do.call(fastai2$callback$hook$Hooks, args)

}


#' @title Hook_output
#'
#' @description Return a `Hook` that stores activations of `module` in `self$stored`
#'
#'
#' @param module module
#' @param detach detach or not
#' @param cpu cpu or not
#' @param grad grad or not
#' @return None
#' @export
hook_output <- function(module, detach = TRUE, cpu = FALSE, grad = FALSE) {

  args <- list(
    module = module,
    detach = detach,
    cpu = cpu,
    grad = grad
  )

  do.call(fastai2$callback$hook$hook_output, args)

}


#' @title Hook_outputs
#'
#' @description Return `Hooks` that store activations of all `modules` in `self.stored`
#'
#'
#' @param modules modules
#' @param detach detach or not
#' @param cpu cpu or not
#' @param grad grad or not
#' @return None
#' @export
hook_outputs <- function(modules, detach = TRUE, cpu = FALSE, grad = FALSE) {

  args <- list(
    modules = modules,
    detach = detach,
    cpu = cpu,
    grad = grad
  )

  do.call(fastai2$callback$hook$hook_outputs, args)

}

######################## model summary #########################################

#' @title Dummy_eval
#'
#' @description Evaluate `m` on a dummy input of a certain `size`
#'
#'
#' @param m m parameter
#' @param size size
#' @return None
#' @export
dummy_eval <- function(m, size = list(64, 64)) {

  fastai2$callback$hook$dummy_eval(
    m = m,
    size = as.list(as.integer(unlist(size)))
  )

}



#' @title Model_sizes
#'
#' @description Pass a dummy input through the model `m` to get the various sizes of activations.
#'
#'
#' @param m m parameter
#' @param size size
#' @return None
#' @export
model_sizes <- function(m, size = list(64, 64)) {

  fastai2$callback$hook$model_sizes(
    m = m,
    size = as.list(as.integer(unlist(size)))
  )

}


#' @title Num_features_model
#'
#' @description Return the number of output features for `m`.
#'
#'
#' @param m m parameter
#' @return None
#' @export
num_features_model <- function(m) {

  fastai2$callback$hook$num_features_model(
    m = m
  )

}


#' @title Has_params
#'
#' @description Check if `m` has at least one parameter
#'
#'
#' @param m m parameter
#' @return None
#' @export
has_params <- function(m) {

  fastai2$callback$hook$has_params(
    m = m
  )

}

#' @title Total_params
#'
#' @description Give the number of parameters of a module and if it's trainable or not
#'
#'
#' @param m m parameter
#' @return None
#' @export
total_params <- function(m) {

  fastai2$callback$hook$total_params(
    m = m
  )

}

#' @title Layer_info
#'
#' @description Return layer infos of `model` on `xb` (only support batch first inputs)
#'
#'
#' @param learn learner/model
#' @param ... additional arguments
#' @return None
#' @export
layer_info <- function(learn, ...) {

  fastai2$callback$hook$layer_info(
    learn = learn,
    ...
  )

}






