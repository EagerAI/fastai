## ----setup, include=FALSE-----------------------------------------------------
knitr::opts_chunk$set(echo = TRUE, eval = FALSE)

## -----------------------------------------------------------------------------
#  commands_path = "SPEECHCOMMANDS"
#  audio_files = get_audio_files(commands_path)
#  length(audio_files)

## -----------------------------------------------------------------------------
#  DBMelSpec = SpectrogramTransformer(mel=TRUE, to_db=TRUE)
#  a2s = DBMelSpec()
#  crop_4000ms = CropSignal(4000)
#  tfms = list(crop_4000ms, a2s)

## -----------------------------------------------------------------------------
#  auds = DataBlock(blocks = list(AudioBlock(), CategoryBlock()),
#                   get_items = get_audio_files,
#                   splitter = RandomSplitter(),
#                   item_tfms = tfms,
#                   get_y = parent_label)
#  
#  audio_dbunch = auds %>% dataloaders(commands_path, item_tfms = tfms, bs = 64)

## -----------------------------------------------------------------------------
#  audio_dbunch %>% show_batch(figsize = c(15, 8.5), nrows = 3, ncols = 3, max_n = 9, dpi = 180)

## -----------------------------------------------------------------------------
#  alter_learner = function(learn, channels = 1L) {
#    try(learn$model[0][0][['in_channels']] <- channels,
#        silent = TRUE)
#    try(learn$model[0][0][['weight']] <- torch$nn$parameter$Parameter(torch$narrow(learn$model[0][0][['weight']],1L,1L,1L)),
#        silent = TRUE)
#  }
#  
#  
#  learn = Learner(dls, xresnet18(pretrained = FALSE), nn$CrossEntropyLoss(), metrics=accuracy)
#  
#  nnchannels = dls %>% one_batch() %>% .[[1]] %>% .$shape %>% .[1]
#  
#  alter_learner(learn, nnchannels)

## -----------------------------------------------------------------------------
#  learn %>% fit_one_cycle(5, lr_max=slice(1e-2))

