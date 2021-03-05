## ----setup, include=FALSE-----------------------------------------------------
knitr::opts_chunk$set(echo = TRUE, eval = FALSE)

## -----------------------------------------------------------------------------
#  library(rBayesianOptimization)
#  library(magrittr)
#  library(fastai)
#  
#  df = data.table::fread('train.csv')
#  df$ID_code <- NULL
#  df$target <- as.character(df$target)
#  
#  procs = list(FillMissing(),Categorify(),Normalize())
#  
#  pct_80 = round(nrow(df) * .8)
#  
#  dep_var = 'target'
#  cont_names = setdiff(names(df), dep_var)
#  
#  dls = TabularDataTable(df, procs, NULL, cont_names,
#                         y_names = dep_var, splits = list(c(1:pct_80),c(c(pct_80+1):nrow(df))
#                                                          )) %>%
#    dataloaders(bs = 100)
#  
#  fastai_fit = function(layer_1, layer_2, layer_3, lr, wd, emb_p) {
#    model <- dls %>% tabular_learner(layers = c(layer_1, layer_2, layer_3),
#                                    wd = wd, config = tabular_config(embed_p = emb_p,
#                                                                     use_bn = TRUE),
#                                    metrics=list(RocAucBinary(),accuracy()),
#                                    cbs = list(EarlyStoppingCallback(monitor='valid_loss',
#                                                                     patience = 2))
#                                    )
#  
#    result_ <- model %>% fit_one_cycle(10,lr)
#  
#    score_ <- list(Score = unlist(tail(result_$roc_auc_score,1)),
#                   Pred = 0)
#    rm(model)
#  
#    return(score_)
#  }
#  
#  search_bound_fastai <- list(layer_1 = c(20,200), layer_2 = c(20,200),
#                              layer_3 = c(20,200),
#                              lr = c(0, 0.1), wd = c(0, 0.1),
#                              emb_p = c(0,1)
#                             )
#  set.seed(123)
#  search_grid_fastai <- data.frame(layer_1 = runif(30, 20, 200),
#                                  layer_2 = runif(30, 20, 200),
#                                  layer_3 = runif(30, 20, 200),
#                                  lr = runif(30, 0, 0.1),
#                                  wd = runif(30, 0, 0.1),
#                                  emb_p = runif(30, 0, 1)
#                                  )
#  head(search_grid_fastai)
#  
#  set.seed(123)
#  bayes_fastai <- BayesianOptimization(FUN = fastai_fit, bounds = search_bound_fastai,
#                                      init_points = 2, init_grid_dt = search_grid_fastai,
#                                      n_iter = 5, acq = "ucb")
#  
#  
#  bayes_fastai$Best_Par

