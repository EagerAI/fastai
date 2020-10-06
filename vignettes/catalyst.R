## ----setup, include=FALSE-----------------------------------------------------
knitr::opts_chunk$set(echo = TRUE,eval = FALSE,echo = T)

## -----------------------------------------------------------------------------
#  library(fastai)
#  library(magrittr)
#  
#  loaders = loaders()
#  
#  data = Data_Loaders(loaders['train'], loaders['valid'])$cuda()
#  
#  model = nn$Sequential() +
#    nn$Flatten() +
#    nn$Linear(28L * 28L, 10L)

## -----------------------------------------------------------------------------
#  metrics = list(accuracy,top_k_accuracy)
#  learn = Learner(data, model, loss_func = F$cross_entropy, opt_func = Adam,
#                  metrics = metrics)
#  
#  learn %>% fit_one_cycle(1, 0.02)

