## ----setup, include=FALSE-----------------------------------------------------
knitr::opts_chunk$set(echo = TRUE,eval = FALSE,echo = T)

## -----------------------------------------------------------------------------
#  library(magrittr)
#  library(fastai)
#  df = data.table::fread('datasets_236694_503227_HR_comma_sep.csv')
#  str(df)

## -----------------------------------------------------------------------------
#  df[['left']] = as.factor(df[['left']])

## -----------------------------------------------------------------------------
#  dep_var = 'left'
#  cat_names = c('sales', 'salary')
#  cont_names = c("satisfaction_level", "last_evaluation", "number_project",
#                 "average_montly_hours", "time_spend_company",
#                 "Work_accident", "promotion_last_5years")

## -----------------------------------------------------------------------------
#  tot = 1:nrow(df)
#  tr_idx = sample(nrow(df), 0.8 * nrow(df))
#  ts_idx = tot[!tot %in% tr_idx]

## -----------------------------------------------------------------------------
#  procs = list(FillMissing(),Categorify(),Normalize())

## -----------------------------------------------------------------------------
#  dls = TabularDataTable(df, procs, cat_names, cont_names,
#                         y_names = dep_var, splits = list(tr_idx, ts_idx) ) %>%
#    dataloaders(bs = 50)

## -----------------------------------------------------------------------------
#  model = dls %>% tabular_learner(layers=c(200,100,100,200),
#                                  config = tabular_config(embed_p = 0.3, use_bn = FALSE),
#                                  metrics = list(accuracy, RocAucBinary(),
#                                               Precision(), Recall(),
#                                               F1Score()))

## -----------------------------------------------------------------------------
#  model %>% lr_find()
#  # SuggestedLRs(lr_min=0.002754228748381138, lr_steep=1.5848931980144698e-06)
#  
#  model %>% plot_lr_find()

## -----------------------------------------------------------------------------
#  res = model %>% fit(5, lr = 0.005)

## -----------------------------------------------------------------------------
#  model %>% get_confusion_matrix() %>%
#    fourfoldplot(conf.level = 0, color = c("#ed3b3b", "#0099ff"),
#               margin = 1,main = paste("Confusion Matrix",
#                                       round(sum(diag(.))/sum(.)*100,0),"%",sep = ' '))

## -----------------------------------------------------------------------------
#  model %>% predict(df[1000:1010,])

