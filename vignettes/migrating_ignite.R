## ----setup, include=FALSE-----------------------------------------------------
knitr::opts_chunk$set(echo = TRUE,eval = FALSE,echo = T)

## -----------------------------------------------------------------------------
#  library(fastai)
#  library(magrittr)
#  
#  data = Data_Loaders(get_data_loaders(64, 128))$cuda()
#  
#  opt_func = partial(SGD, momentum=0.5)
#  learn = Learner(data, Net(), loss_func = nn$NLLLoss(), opt_func = opt_func, metrics = accuracy)
#  learn %>% fit_one_cycle(1, 0.01)

