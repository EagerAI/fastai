## ----setup, include=FALSE-----------------------------------------------------
knitr::opts_chunk$set(echo = TRUE,eval = FALSE,echo = T)

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

## -----------------------------------------------------------------------------
#  params = L(lapply(0:3, function(x) tst_param(x)))
#  
#  opt = Optimizer(params, sgd_step, lr=0.1)
#  
#  try(params[3]$grad <- NULL,
#      TRUE)
#  
#  params[3]$grad
#  
#  opt$step()
#  
#  str(params$items)

## -----------------------------------------------------------------------------
#  params = L(lapply(0:3, function(x) tst_param(x)))
#  
#  opt = Optimizer(list(params[0:1],params[2:3]), sgd_step, lr=0.1)
#  
#  opt$hypers$items[[1]][[1]] = 0.01
#  
#  opt$step()
#  
#  str(params$items)

## -----------------------------------------------------------------------------
#  params = L(lapply(0:3, function(x) tst_param(x)))
#  
#  opt = Optimizer(params, list(weight_decay, sgd_step), lr=0.1, wd = 0.1)
#  
#  opt$zero_grad()
#  
#  str(params$items)

## -----------------------------------------------------------------------------
#  p = tst_param(c(1,2,3), c(4,5,6))
#  state = average_grad(p, mom = 0.9, dampening = FALSE, grad_avg = NULL)
#  p$grad
#  # tensor([4., 5., 6.])
#  
#  state = average_grad(p, mom=0.9, dampening = TRUE)
#  p$grad*0.1
#  # tensor([0.4000, 0.5000, 0.6000])
#  p$grad*(0.1*0.9+0.1)
#  # tensor([0.7600, 0.9500, 1.1400])

## -----------------------------------------------------------------------------
#  p = tst_param(c(1,2,3), c(4,5,6))
#  state = average_sqr_grad(p, sqr_mom = 0.99, dampening = FALSE)
#  
#  p$grad$pow(2)
#  # tensor([16., 25., 36.])
#  
#  p$grad$pow(2) * 1.99
#  # tensor([31.8400, 49.7500, 71.6400])
#  
#  state = average_sqr_grad(p, sqr_mom = 0.99)
#  p$grad$pow(2) * 1e-2
#  # tensor([0.1600, 0.2500, 0.3600])
#  state = average_sqr_grad(p, sqr_mom = 0.99)
#  
#  p$grad$pow(2)*(0.01*0.99+0.01)
#  # tensor([0.3184, 0.4975, 0.7164])
#  
#  params = L(lapply(0:3, function(x) tst_param(x)))
#  opt = Optimizer(params, sgd_step, lr = 0.1)
#  opt$freeze_to(1L)

## -----------------------------------------------------------------------------
#  params = L(lapply(0:3, function(x) tst_param(x)))
#  opt = SGD(params, lr = 0.1)
#  opt$step()
#  str(params$items)

## -----------------------------------------------------------------------------
#  params = L(lapply(0:3, function(x) tst_param(x)))
#  opt = SGD(params, lr = 0.1, mom = 0.9)
#  opt$step()
#  str(params$items)

## -----------------------------------------------------------------------------
#  params =  L(lapply(0:3, function(x) tst_param(x)))
#  #Weight decay
#  opt = SGD(params, lr=0.1, mom=0.9, wd=0.1)
#  opt$step()
#  str(params$items)

## -----------------------------------------------------------------------------
#  params =  L(lapply(0:3, function(x) tst_param(x)))
#  #L2 reg
#  opt = SGD(params, lr=0.1, mom=0.9, wd=0.1, decouple_wd=FALSE)
#  opt$step()
#  str(params$items)

## -----------------------------------------------------------------------------
#  params = tst_param(c(1:3), c(0.1,0.2,0.3))
#  opt = RMSProp(params, lr=0.1)
#  opt$step()
#  opt$step()
#  step = (-0.1 * 0.1) / (sqrt((0.01*0.99+0.01) * 0.1**2) + 1e-8)
#  params; tensor(c(step, 1+step, 2+step))

## -----------------------------------------------------------------------------
#  params = tst_param(c(1:3), c(0.1,0.2,0.3))
#  opt = RMSProp(params, lr=0.1, mom=0.9)
#  opt$step()
#  opt$step()
#  step = (- 0.1 * (0.1 + 0.9*0.1)) / (sqrt((0.01*0.99+0.01) * 0.1**2) + 1e-8)
#  params; tensor(c(step, 1+step, 2+step))

## -----------------------------------------------------------------------------
#  params = tst_param(c(1:3), c(0.1,0.2,0.3))
#  opt = Adam(params, lr=0.1, wd=0)
#  opt$step()
#  step = (-0.1 * 0.1) / (sqrt(0.1**2) + 1e-8)
#  params; tensor(c(1+step, 2+step, 3+step))

## -----------------------------------------------------------------------------
#  opt$step()
#  params;tensor(tensor(c(1+2*step, 2+2*step, 3+2*step)))

## -----------------------------------------------------------------------------
#  beta = 0.99
#  r_inf = 2/(1-beta) - 1
#  rs = lapply(5:500, function(s) {r_inf - 2*s*beta**s/(1-beta**s)}) %>% as.numeric()
#  v = sqrt(((rs-4) * (rs-2) * r_inf)/((r_inf-4)*(r_inf-2)*rs))
#  df_high = data.frame(x = 1:length(v), y = v)
#  
#  library(highcharter)
#  hchart(df_high,'line', hcaes(x,y))

## -----------------------------------------------------------------------------
#  params = tst_param(c(1:3), c(0.1,0.2,0.3))
#  opt = QHAdam(params, lr=0.1)
#  opt$step()
#  step = (-0.1 * (((1-0.7) * 0.1) + (0.7 * 0.1)) )/ (
#   sqrt(((1-1.0) * 0.1**2) + (1.0 * 0.1**2)) + 1e-8)
#  params; tensor(c(1+step, 2+step, 3+step))
#  # tensor([0.9000, 1.9000, 2.9000])
#  # tensor([0.9000, 1.9000, 2.9000])
#  opt$step()
#  params; tensor(c(1+2*step, 2+2*step, 3+2*step))
#  # tensor([0.8000, 1.8000, 2.8000])
#  # tensor([0.8000, 1.8000, 2.8000])

## -----------------------------------------------------------------------------
#  params = list(tst_param(c(1:3), c(0.1,0.2,0.3)), tst_param(c(1:3), c(0.01,0.02,0.03)))
#  opt = Larc(params, lr=0.1)
#  opt$step()
#  #First param local lr is 0.02 < lr so it's not clipped
#  opt$state[params[[1]]]['local_lr']

## -----------------------------------------------------------------------------
#  opt$state[params[[2]]]['local_lr']

## -----------------------------------------------------------------------------
#  params = list(tst_param(c(1:3), c(0.1,0.2,0.3)), tst_param(c(1:3), c(0.01,0.02,0.03)))
#  opt = Larc(params, lr=0.1, clip = FALSE)
#  opt$step()
#  #Second param local lr is 0.2 > lr so it's clipped
#  opt$state[params[[1]]]['local_lr']

## -----------------------------------------------------------------------------
#  opt$state[params[[2]]]['local_lr']

## -----------------------------------------------------------------------------
#  params = tst_param(c(1:3), c(0.1,0.2,0.3))
#  opt = Lamb(params, lr=0.1)
#  opt$step()
#  params

## -----------------------------------------------------------------------------
#  params = tst_param(c(1:3), c(0.1,0.2,0.3))
#  p = params$data$clone()
#  g = tensor(c(0.1,0.2,0.3))
#  opt = Lookahead(SGD(params, lr=0.1))
#  
#  for(i in 1:5) {
#    opt$step()
#  }
#  #first 5 steps are normal SGD steps
#  params; p - g * 0.5
#  # tensor([0.9500, 1.9000, 2.8500])
#  # tensor([0.9500, 1.9000, 2.8500])
#  
#  #Since k=6, sixth step is a moving average of the 6 SGD steps with the initial weight
#  opt$step()
#  params; p * 0.5 + (p-g*0.6) * 0.5
#  # tensor([0.9700, 1.9400, 2.9100])
#  # tensor([0.9700, 1.9400, 2.9100])

