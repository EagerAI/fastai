## ----setup, include=FALSE-----------------------------------------------------
knitr::opts_chunk$set(echo = TRUE, eval = FALSE)

## -----------------------------------------------------------------------------
#  library(fastai)
#  library(magrittr)
#  library(zeallot)
#  URLs_IMDB_SAMPLE()

## -----------------------------------------------------------------------------
#  HF_TASKS_AUTO = HF_TASKS_AUTO()
#  task = HF_TASKS_AUTO$SequenceClassification
#  
#  pretrained_model_name = "roberta-base" # "distilbert-base-uncased" "bert-base-uncased"
#  c(hf_arch, hf_config, hf_tokenizer, hf_model) %<-% get_hf_objects(pretrained_model_name, task=task)

## -----------------------------------------------------------------------------
#  imdb_df = data.table::fread('imdb_sample/texts.csv')
#  
#  blocks = list(HF_TextBlock(hf_arch=hf_arch, hf_tokenizer=hf_tokenizer), CategoryBlock())
#  
#  dblock = DataBlock(blocks=blocks,
#                     get_x=ColReader('text'),
#                     get_y=ColReader('label'),
#                     splitter=ColSplitter(col='is_valid'))
#  
#  dls = dblock %>% dataloaders(imdb_df, bs=4)
#  dls %>% one_batch()

## -----------------------------------------------------------------------------
#  model = HF_BaseModelWrapper(hf_model)
#  
#  learn = Learner(dls,
#                  model,
#                  opt_func=partial(Adam, decouple_wd=TRUE),
#                  loss_func=CrossEntropyLossFlat(),
#                  metrics=accuracy,
#                  cbs=HF_BaseModelCallback(),
#                  splitter=hf_splitter())
#  
#  learn$create_opt()
#  learn$freeze()
#  
#  learn %>% summary()

## -----------------------------------------------------------------------------
#  result = learn %>% fit_one_cycle(3, lr_max=1e-3)
#  
#  learn %>% predict(imdb_df$text[1:4])

