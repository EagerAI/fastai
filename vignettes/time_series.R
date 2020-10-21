## ----setup, include=FALSE-----------------------------------------------------
knitr::opts_chunk$set(echo = TRUE,eval = FALSE,echo = T)

## -----------------------------------------------------------------------------
#  
#  library(dplyr)
#  library(fastai)
#  
#  df = data.table::fread('https://raw.githubusercontent.com/facebook/prophet/master/examples/example_wp_log_peyton_manning.csv')

## -----------------------------------------------------------------------------
#  split_idx = which(df$ds=='2016-01-01') # take 1 year for validation
#  
#  y = df$y
#  
#  df = timetk::tk_augment_timeseries_signature(df) %>%
#    mutate_if(is.factor,  as.numeric) %>%
#    select(-ds, -hour, -minute, -second, -hour12, -am.pm, -y) %>%
#    scale() %>% data.table::as.data.table()
#  
#  df[is.na(df)]=0
#  df$y = y

## -----------------------------------------------------------------------------
#  df_train = df[1:split_idx,]
#  df_test = df[(split_idx+1):nrow(df),]
#  
#  x_cols = setdiff(colnames(df_train),'y')

## -----------------------------------------------------------------------------
#  dls = TSDataLoaders_from_dfs(df_train, df_test, x_cols = x_cols, label_col = 'y', bs=60,
#                               y_block = RegressionBlock())
#  
#  dls %>% show_batch()
#  
#  inception = create_inception(1, 1)
#  
#  learn = Learner(dls, inception, metrics=list(mae(), rmse()))

## -----------------------------------------------------------------------------
#  lrs = learn %>% lr_find()
#  
#  learn %>% plot_lr_find()

## -----------------------------------------------------------------------------
#  learn %>% fit_one_cycle(30, 1e-5, cbs = EarlyStoppingCallback(patience = 5))
#  
#  learn %>% predict(df_test)
#  
#  # to R
#  # result = learn %>% predict(df_test)
#  # result$cpu()$numpy()

