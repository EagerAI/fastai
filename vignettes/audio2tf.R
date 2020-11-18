## ----setup, include=FALSE-----------------------------------------------------
knitr::opts_chunk$set(echo = TRUE, eval = FALSE)

## -----------------------------------------------------------------------------
#  commands_path = "SPEECHCOMMANDS"
#  audio_files = get_audio_files(commands_path)
#  length(audio_files$items)
#  # [1] 105835

## -----------------------------------------------------------------------------
#  DBMelSpec = SpectrogramTransformer(mel=TRUE, to_db=TRUE)
#  a2s = DBMelSpec()
#  crop_4000ms = ResizeSignal(4000)
#  tfms = list(crop_4000ms, a2s)

## -----------------------------------------------------------------------------
#  auds = DataBlock(blocks = list(AudioBlock(), CategoryBlock()),
#                   get_items = get_audio_files,
#                   splitter = RandomSplitter(),
#                   item_tfms = tfms,
#                   get_y = parent_label)
#  
#  audio_dbunch = auds %>% dataloaders(commands_path, item_tfms = tfms, bs = 20)

## -----------------------------------------------------------------------------
#  audio_dbunch %>% show_batch(figsize = c(15, 8.5), nrows = 3, ncols = 3, max_n = 9, dpi = 180)

## -----------------------------------------------------------------------------
#  torch = torch()
#  nn = nn()
#  
#  # channel from 3 to 1
#  learn$model[0][0][['in_channels']] %f% 1L
#  # reshape
#  new_weight_shape <- torch$nn$parameter$Parameter(
#    (learn$model[0][0]$weight %>% narrow('[:,1,:,:]'))$unsqueeze(1L))
#  
#  # assign with %f%
#  learn$model[0][0][['weight']] %f% new_weight_shape

## -----------------------------------------------------------------------------
#  # login for the 1st time then remove it
#  login("API_key_from_wandb_dot_ai")
#  init(project='R')

## -----------------------------------------------------------------------------
#  learn %>% fit_one_cycle(3, lr_max=slice(1e-2), cbs = list(WandbCallback()))

