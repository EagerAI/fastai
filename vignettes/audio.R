## ----setup, include=FALSE-----------------------------------------------------
knitr::opts_chunk$set(echo = TRUE, eval = FALSE)

## -----------------------------------------------------------------------------
#  URLs_SPEAKERS10()
#  path_dig = 'SPEAKERS10'

## -----------------------------------------------------------------------------
#  audio_extensions()[1:6]
#  #[1] ".aif"  ".aifc" ".aiff" ".au"   ".m3u"  ".mp2"

## -----------------------------------------------------------------------------
#  fnames = get_files(path_dig, extensions = audio_extensions())
#  # (#3842) [Path('SPEAKERS10/f0004_us_f0004_00414.wav')...]

## -----------------------------------------------------------------------------
#  at = AudioTensor_create(fnames[0])
#  at; at$shape
#  at %>% show() %>% plot(dpi = 200)

## -----------------------------------------------------------------------------
#  cfg = Voice()
#  
#  cfg$f_max; cfg$sample_rate
#  #[1] 8000 # frequency range
#  #[1] 16000 # the sampling rate

## -----------------------------------------------------------------------------
#  aud2spec = AudioToSpec_from_cfg(cfg)
#  
#  crop1s = ResizeSignal(1000)

## -----------------------------------------------------------------------------
#  pipe = Pipeline(list(AudioTensor_create, crop1s, aud2spec))
#  pipe(fnames[0]) %>% show() %>% plot(dpi = 200)

## -----------------------------------------------------------------------------
#  item_tfms = list(ResizeSignal(1000), aud2spec)
#  
#  get_y = function(x) substring(x$name[1],1,1)
#  
#  aud_digit = DataBlock(blocks = list(AudioBlock(), CategoryBlock()),
#                        get_items = get_audio_files,
#                        splitter = RandomSplitter(),
#                        item_tfms = item_tfms,
#                        get_y = get_y)
#  
#  dls = aud_digit %>% dataloaders(source = path_dig, bs = 64)
#  
#  dls %>% show_batch(figsize = c(15, 8.5), nrows = 3, ncols = 3, max_n = 9, dpi = 180)

## -----------------------------------------------------------------------------
#  torch = torch()
#  nn = nn()
#  
#  learn = Learner(dls, xresnet18(pretrained = FALSE), nn$CrossEntropyLoss(), metrics=accuracy)
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
#  lrs = learn %>% lr_find()
#  #SuggestedLRs(lr_min=0.03019951581954956, lr_steep=0.0030199517495930195)

## -----------------------------------------------------------------------------
#  learn %>% fit_one_cycle(10, 1e-3)

