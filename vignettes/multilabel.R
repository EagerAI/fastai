## ----setup, include=FALSE-----------------------------------------------------
knitr::opts_chunk$set(echo = TRUE, eval = FALSE)

## -----------------------------------------------------------------------------
#  library(fastai)
#  library(magrittr)
#  library(zeallot)
#  df = HF_load_dataset('civil_comments', split='train[:1%]')

## -----------------------------------------------------------------------------
#  df = data.table::as.data.table(df)
#  
#  lbl_cols = c('severe_toxicity',
#               'obscene',
#               'threat',
#               'insult',
#               'identity_attack',
#               'sexual_explicit')
#  
#  df <- df[,(lbl_cols) := round(.SD,0), .SDcols=lbl_cols]
#  df <- df[, (lbl_cols) := lapply(.SD, as.integer), .SDcols=lbl_cols]

## -----------------------------------------------------------------------------
#  task = HF_TASKS_ALL()$SequenceClassification
#  
#  pretrained_model_name = "distilroberta-base"
#  config = AutoConfig()$from_pretrained(pretrained_model_name)
#  config$num_labels = length(lbl_cols)
#  
#  c(hf_arch, hf_config, hf_tokenizer, hf_model) %<-% get_hf_objects(pretrained_model_name,
#                                                                                 task=task,
#                                                                                 config=config)

## -----------------------------------------------------------------------------
#  blocks = list(
#    HF_TextBlock(hf_arch=hf_arch, hf_tokenizer=hf_tokenizer),
#    MultiCategoryBlock(encoded=TRUE, vocab=lbl_cols)
#  )
#  
#  dblock = DataBlock(blocks=blocks,
#                     get_x=ColReader('text'), get_y=ColReader(lbl_cols),
#                     splitter=RandomSplitter())
#  
#  dls = dblock %>% dataloaders(df, bs=8)
#  
#  dls %>% one_batch()

## -----------------------------------------------------------------------------
#  model = HF_BaseModelWrapper(hf_model)
#  
#  learn = Learner(dls,
#                  model,
#                  opt_func=partial(Adam),
#                  loss_func=BCEWithLogitsLossFlat(),
#                  metrics=partial(accuracy_multi(), thresh=0.2),
#                  cbs=HF_BaseModelCallback(),
#                  splitter=hf_splitter())
#  
#  learn$loss_func$thresh = 0.2
#  learn$create_opt()             # -> will create your layer groups based on your "splitter" function
#  learn$freeze()
#  
#  learn %>% summary()

## -----------------------------------------------------------------------------
#  lrs = learn %>% lr_find(suggestions=TRUE)
#  
#  learn %>% fit_one_cycle(1, lr_max=1e-2)

## -----------------------------------------------------------------------------
#  learn$loss_func$thresh = 0.02
#  
#  learn %>% predict("Those damned affluent white people should only eat their own food, like cod cakes and boiled potatoes.
#  No enchiladas for them!")

