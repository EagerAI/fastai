## ----setup, include=FALSE-----------------------------------------------------
knitr::opts_chunk$set(echo = TRUE,eval = FALSE)

## -----------------------------------------------------------------------------
#  library(magrittr)
#  library(fastai)
#  
#  tst_param = function(val, grad = NULL) {
#    "Create a tensor with `val` and a gradient of `grad` for testing"
#    res = tensor(val) %>% float()
#  
#    if(is.null(grad)) {
#      grad = tensor(val / 10)
#    } else {
#      grad = tensor(grad)
#    }
#  
#    res$grad = grad %>% float()
#    res
#  }

## -----------------------------------------------------------------------------
#  p = tst_param(1., 0.1)
#  p

## -----------------------------------------------------------------------------
#  sgd_step(p, 1.)
#  p

## -----------------------------------------------------------------------------
#  p$grad

## -----------------------------------------------------------------------------
#  p = tst_param(1., 0.1)
#  weight_decay(p, 1., 0.1)
#  p

## -----------------------------------------------------------------------------
#  p = tst_param(1., 0.1)
#  l2_reg(p, 1., 0.1)
#  p$grad

## -----------------------------------------------------------------------------
#  params = L(lapply(0:3, function(x) tst_param(x)))
#  
#  opt = Optimizer(params, sgd_step, lr=0.1)
#  
#  opt$step()
#  
#  str(params$items)

## -----------------------------------------------------------------------------
#  params = L(lapply(0:3, function(x) tst_param(x)))
#  
#  opt = Optimizer(params, list(weight_decay, sgd_step), lr=0.1, wd = 0.1)
#  
#  opt$step()
#  
#  str(params$items)

