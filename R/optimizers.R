#' @title Adam
#'
#'
#' @param ... parameters to pass
#' @return None
#' @export
Adam <- function(...) {
  args = list(...)

  if(is.null(unlist(args))) {
    vision$all$Adam
  } else {
    do.call(vision$all$Adam, args)
  }

}

#attr(Adam ,"py_function_name") <- "Adam"


#' @title RAdam
#'
#'
#' @param ... parameters to pass
#' @return None
#' @export
RAdam <- function(...) {
  args = list(...)

  if(is.null(unlist(args))) {
    vision$all$RAdam
  } else {
    do.call(vision$all$RAdam, args)
  }

}

#attr(RAdam ,"py_function_name") <- "RAdam"

#' @title SGD
#'
#'
#' @param ... parameters to pass
#' @return None
#' @export
SGD <- function(...) {
  args = list(...)

  if(is.null(unlist(args))) {
    vision$all$SGD
  } else {
    do.call(vision$all$SGD, args)
  }

}

#attr(SGD ,"py_function_name") <- "SGD"


#' @title RMSProp
#'
#'
#' @param ... parameters to pass
#' @return None
#' @export
RMSProp <- function(...) {
  args = list(...)

  if(is.null(unlist(args))) {
    vision$all$RMSProp
  } else {
    do.call(vision$all$RMSProp, args)
  }

}

#attr(RMSProp ,"py_function_name") <- "RMSProp"


#' @title QHAdam
#'
#'
#' @param ... parameters to pass
#' @return None
#' @export
QHAdam <- function(...) {
  args = list(...)

  if(is.null(unlist(args))) {
    vision$all$QHAdam
  } else {
    do.call(vision$all$QHAdam, args)
  }

}

#attr(QHAdam ,"py_function_name") <- "QHAdam"


#' @title Larc
#'
#'
#' @param ... parameters to pass
#' @return None
#' @export
Larc <- function(...) {
  args = list(...)

  if(is.null(unlist(args))) {
    vision$all$Larc
  } else {
    do.call(vision$all$Larc, args)
  }

}

#' @title Lamb
#'
#'
#' @param ... parameters to pass
#' @return None
#' @export
Lamb <- function(...) {
  args = list(...)

  if(is.null(unlist(args))) {
    vision$all$Lamb
  } else {
    do.call(vision$all$Lamb, args)
  }

}



#' @title Lookahead
#'
#'
#' @param ... parameters to pass
#' @return None
#' @export
Lookahead <- function(...) {
  args = list(...)

  if(is.null(unlist(args))) {
    vision$all$Lookahead
  } else {
    do.call(vision$all$Lookahead, args)
  }

}


#' @title OptimWrapper
#'
#'
#' @param ... parameters to pass
#' @return None
#' @export
OptimWrapper <- function(...) {
  args = list(...)

  if(is.null(unlist(args))) {
    vision$all$OptimWrapper
  } else {
    do.call(vision$all$OptimWrapper, args)
  }

}


#' @title Optimizer
#'
#'
#' @param ... parameters to pass
#' @return None
#' @export
Optimizer <- function(...) {
  args = list(...)

  if(is.null(unlist(args))) {
    vision$all$Optimizer
  } else {
    do.call(vision$all$Optimizer, args)
  }

}


#' @title Noop
#'
#'
#' @param ... parameters to pass
#' @return None
#' @export
noop <- function(...) {

  args = list(...)
  if(length(args)>0) {
    vision$all$noop(
      x = x
    )
  } else {
    vision$all$noop
  }

}


#' @title Sgd_step
#'
#' @return None
#' @param p p
#' @param lr learning rate
#' @param ... additional arguments to pass
#' @export
sgd_step <- function(p, lr, ...) {

  args <- list(
    p = p,
    lr = lr,
    ...
  )

  do.call(vision$all$sgd_step, args)

}

#' @title Weight_decay
#'
#' @description Weight decay as decaying `p` with `lr*wd`
#'
#'
#' @param p p
#' @param lr learning rate
#' @param wd weight decay
#' @param do_wd do_wd
#' @param ... additional args to pass
#' @return None
#' @export
weight_decay <- function(p, lr, wd, do_wd = TRUE, ...) {

  args <- list(
    p = p,
    lr = lr,
    wd = wd,
    do_wd = do_wd,
    ...
  )

  do.call(vision$all$weight_decay, args)

}


#' @title L2_reg
#'
#' @description L2 regularization as adding `wd*p` to `p$grad`
#'
#'
#' @param p p
#' @param lr learning rate
#' @param wd weight decay
#' @param do_wd do_wd
#' @param ... additional arguments to pass
#' @return None
#' @export
l2_reg <- function(p, lr, wd, do_wd = TRUE, ...) {

  args <- list(
    p = p,
    lr = lr,
    wd = wd,
    do_wd = do_wd,
    ...
  )

  do.call(vision$all$l2_reg, args)

}


#' @title Average_grad
#'
#' @description Keeps track of the avg grads of `p` in `state` with `mom`.
#'
#'
#' @param p p
#' @param mom momentum
#' @param dampening dampening
#' @param grad_avg grad average
#' @param ... additional args to pass
#' @return None
#' @export
average_grad <- function(p, mom, dampening = FALSE, grad_avg = NULL, ...) {

  args <- list(
    p = p,
    mom = mom,
    dampening = dampening,
    grad_avg = grad_avg,
    ...
  )

  do.call(vision$all$average_grad, args)

}


#' @title Average_sqr_grad
#'
#'
#' @param p p
#' @param sqr_mom sqr momentum
#' @param dampening dampening
#' @param sqr_avg sqr average
#' @param ... additional args to pass
#' @return None
#' @export
average_sqr_grad <- function(p, sqr_mom, dampening = TRUE, sqr_avg = NULL, ...) {

  args <- list(
    p = p,
    sqr_mom = sqr_mom,
    dampening = dampening,
    sqr_avg = sqr_avg,
    ...
  )

  do.call(vision$all$average_sqr_grad, args)

}


#' @title Momentum_step
#'
#' @description Step for SGD with momentum with `lr`
#'
#'
#' @param p p
#' @param lr learning rate
#' @param grad_avg grad average
#' @param ... additional arguments to pass
#' @return None
#' @export
momentum_step <- function(p, lr, grad_avg, ...) {

  args <- list(
    p = p,
    lr = lr,
    grad_avg = grad_avg,
    ...
  )

  do.call(vision$all$momentum_step, args)

}


#' @title Rms_prop_step
#'
#' @description Step for SGD with momentum with `lr`
#'
#'
#' @param p p
#' @param lr learning rate
#' @param sqr_avg sqr average
#' @param eps epsilon
#' @param grad_avg grad average
#' @param ... additional arguments to pass
#' @return None
#' @export
rms_prop_step <- function(p, lr, sqr_avg, eps, grad_avg = NULL, ...) {

  args <- list(
    p = p,
    lr = lr,
    sqr_avg = sqr_avg,
    eps = eps,
    grad_avg = grad_avg,
    ...
  )

  do.call(vision$all$rms_prop_step, args)

}



#' @title Step_stat
#'
#' @description Register the number of steps done in `state` for `p`
#'
#' @param ... additional args to pass
#' @param p p
#' @param step step
#' @return None
#' @export
step_stat <- function(p, step = 0, ...) {

  args <- list(
    p = p,
    step = as.integer(step),
    ...
  )

  do.call(vision$all$step_stat, args)

}

#' @title Debias
#'
#'
#' @param mom mom
#' @param damp damp
#' @param step step
#' @return None
#' @export
debias <- function(mom, damp, step) {

  args <- list(
    mom = mom,
    damp = damp,
    step = step
  )

  do.call(vision$all$debias, args)

}


#' @title Adam_step
#'
#' @description Step for Adam with `lr` on `p`
#'
#'
#' @param p p
#' @param lr learning rate
#' @param mom momentum
#' @param step step
#' @param sqr_mom sqr momentum
#' @param grad_avg grad average
#' @param sqr_avg sqr average
#' @param eps epsilon
#' @param ... additional arguments to pass
#' @return None
#' @export
adam_step <- function(p, lr, mom, step, sqr_mom, grad_avg, sqr_avg, eps, ...) {

  args <- list(
    p = p,
    lr = lr,
    mom = mom,
    step = step,
    sqr_mom = sqr_mom,
    grad_avg = grad_avg,
    sqr_avg = sqr_avg,
    eps = eps,
    ...
  )

  do.call(vision$all$adam_step, args)

}


#' @title Radam_step
#'
#' @description Step for RAdam with `lr` on `p`
#'
#'
#' @param p p
#' @param lr learning rate
#' @param mom momentum
#' @param step step
#' @param sqr_mom sqr momentum
#' @param grad_avg grad average
#' @param sqr_avg sqr average
#' @param eps epsilon
#' @param beta beta
#' @param ... additional arguments to pass
#' @return None
#' @export
radam_step <- function(p, lr, mom, step, sqr_mom, grad_avg, sqr_avg, eps, beta, ...) {

  args <- list(
    p = p,
    lr = lr,
    mom = mom,
    step = step,
    sqr_mom = sqr_mom,
    grad_avg = grad_avg,
    sqr_avg = sqr_avg,
    eps = eps,
    beta = beta,
    ...
  )

  do.call(vision$all$radam_step, args)

}

#' @title Qhadam_step
#'
#'
#' @param p p
#' @param lr learning rate
#' @param mom momentum
#' @param sqr_mom sqr momentum
#' @param sqr_avg sqr average
#' @param nu_1 nu_1
#' @param nu_2 nu_2
#' @param step step
#' @param grad_avg gradient average
#' @param eps epsilon
#' @param ... additional arguments to pass
#' @return None
#' @export
qhadam_step <- function(p, lr, mom, sqr_mom, sqr_avg, nu_1, nu_2, step, grad_avg, eps, ...) {

  args <- list(
    p = p,
    lr = lr,
    mom = mom,
    sqr_mom = sqr_mom,
    sqr_avg = sqr_avg,
    nu_1 = nu_1,
    nu_2 = nu_2,
    step = step,
    grad_avg = grad_avg,
    eps = eps,
    ...
  )

  do.call(vision$all$qhadam_step, args)

}


#' @title Larc_layer_lr
#'
#' @description Computes the local lr before weight decay is applied
#'
#'
#' @param p p
#' @param lr learning rate
#' @param trust_coeff trust_coeff
#' @param wd weight decay
#' @param eps epsilon
#' @param clip clip
#' @param ... additional arguments to pass
#' @return None
#' @export
larc_layer_lr <- function(p, lr, trust_coeff, wd, eps, clip = TRUE, ...) {

  args <- list(
    p = p,
    lr = lr,
    trust_coeff = trust_coeff,
    wd = wd,
    eps = eps,
    clip = clip,
    ...
  )

  do.call(vision$all$larc_layer_lr, args)

}


#' @title Larc_step
#'
#' @description Step for LARC `local_lr` on `p`
#'
#'
#' @param p p
#' @param local_lr local learning rate
#' @param grad_avg gradient average
#' @param ... additional args to pass
#' @return None
#' @export
larc_step <- function(p, local_lr, grad_avg = NULL, ...) {

  args <- list(
    p = p,
    local_lr = local_lr,
    grad_avg = grad_avg,
    ...
  )

  do.call(vision$all$larc_step, args)

}

#' @title Lamb_step
#'
#' @description Step for LAMB with `lr` on `p`
#'
#'
#' @param p p
#' @param lr learning rate
#' @param mom momentum
#' @param step step
#' @param sqr_mom sqr momentum
#' @param grad_avg gradient average
#' @param sqr_avg sqr average
#' @param eps epsilon
#' @param ... additional arguments to pass
#' @return None
#' @export
lamb_step <- function(p, lr, mom, step, sqr_mom, grad_avg, sqr_avg, eps, ...) {

  args <- list(
    p = p,
    lr = lr,
    mom = mom,
    step = step,
    sqr_mom = sqr_mom,
    grad_avg = grad_avg,
    sqr_avg = sqr_avg,
    eps = eps,
    ...
  )

  do.call(vision$all$lamb_step, args)

}


#' @title Ranger
#'
#' @description Convenience method for `Lookahead` with `RAdam`
#'
#'
#' @param p p
#' @param lr learning rate
#' @param mom momentum
#' @param wd weight decay
#' @param eps epsilon
#' @param sqr_mom sqr momentum
#' @param beta beta
#' @param decouple_wd decouple weight decay
#' @return None
#' @export
ranger <- function(p, lr, mom = 0.95, wd = 0.01, eps = 1e-06,
                   sqr_mom = 0.99, beta = 0.0, decouple_wd = TRUE) {

  args <- list(
    p = p,
    lr = lr,
    mom = mom,
    wd = wd,
    eps = eps,
    sqr_mom = sqr_mom,
    beta = beta,
    decouple_wd = decouple_wd
  )

  do.call(vision$all$ranger, args)

}


#' @title Detuplify_pg
#'
#'
#' @param d d
#' @return None
#' @export
detuplify_pg <- function(d) {

  vision$all$detuplify_pg(
    d = d
  )

}


#' @title Set_item_pg
#'
#'
#' @param pg pg
#' @param k k
#' @param v v
#' @return None
#' @export
set_item_pg <- function(pg, k, v) {

  vision$all$set_item_pg(
    pg = pg,
    k = k,
    v = v
  )

}









