## ----setup, include=FALSE-----------------------------------------------------
knitr::opts_chunk$set(echo = TRUE,eval = FALSE,echo = T)

## -----------------------------------------------------------------------------
#  URLs_PETS()
#  
#  path = 'oxford-iiit-pet'
#  path_hr = paste(path, 'images', sep = '/')
#  path_lr = paste(path, 'crappy', sep = '/')

## -----------------------------------------------------------------------------
#  # run this only for the first time, then skip
#  items = get_image_files(path_hr)
#  parallel(crappifier(path_lr, path_hr), items)

## -----------------------------------------------------------------------------
#  bs = 10
#  size = 64
#  arch = resnet34
#  
#  get_dls = function(bs, size) {
#    dblock = DataBlock(blocks = list(ImageBlock, ImageBlock),
#                       get_items = get_image_files,
#                       get_y = function(x) {paste(path_hr, as.character(x$name), sep = '/')},
#                       splitter = RandomSplitter(),
#                       item_tfms = Resize(size),
#                       batch_tfms = list(
#                         aug_transforms(max_zoom = 2.),
#                         Normalize_from_stats( imagenet_stats() )
#                       ))
#    dls = dblock %>% dataloaders(path_lr, bs = bs, path = path)
#    dls$c = 3L
#    dls
#  }
#  
#  dls_gen = get_dls(bs, size)

## -----------------------------------------------------------------------------
#  dls_gen %>% show_batch(max_n = 4, dpi = 150)

## -----------------------------------------------------------------------------
#  wd = 1e-3
#  
#  y_range = c(-3.,3.)
#  
#  loss_gen = MSELossFlat()
#  
#  create_gen_learner = function() {
#    unet_learner(dls_gen, arch, loss_func = loss_gen,
#                 config = unet_config(blur=TRUE, norm_type = "Weight",
#                 self_attention = TRUE, y_range = y_range))
#  }
#  
#  
#  learn_gen = create_gen_learner()
#  
#  learn_gen %>% fit_one_cycle(2, pct_start = 0.8, wd = wd)

## -----------------------------------------------------------------------------
#  learn_gen %>% show_results(max_n = 6, dpi = 200)

