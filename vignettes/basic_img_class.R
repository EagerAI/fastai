## ----setup, include=FALSE-----------------------------------------------------
knitr::opts_chunk$set(echo = TRUE,eval = FALSE,echo = T)

## -----------------------------------------------------------------------------
#  URLs_PETS()

## -----------------------------------------------------------------------------
#  path = 'oxford-iiit-pet'
#  path_anno = 'oxford-iiit-pet/annotations'
#  path_img = 'oxford-iiit-pet/images'
#  fnames = get_image_files(path_img)

## -----------------------------------------------------------------------------
#  dls = ImageDataLoaders_from_name_re(
#    path, fnames, pat='(.+)_\\d+.jpg$',
#    item_tfms = RandomResizedCrop(460, min_scale=0.75), bs = 10,
#    batch_tfms = list(aug_transforms(size = 299, max_warp = 0),
#                    Normalize_from_stats( imagenet_stats() )
#    ),
#    device = 'cuda'
#  )

## -----------------------------------------------------------------------------
#  dls %>% show_batch()

## -----------------------------------------------------------------------------
#  learn = cnn_learner(dls, resnet50(), metrics = error_rate)

## -----------------------------------------------------------------------------
#  learn %>% fit_one_cycle(n_epoch = 8)

## -----------------------------------------------------------------------------
#  learn$unfreeze()
#  
#  learn %>% fit_one_cycle(3, lr_max = slice(1e-6,1e-4))

## -----------------------------------------------------------------------------
#  interp = ClassificationInterpretation_from_learner(learn)
#  interp %>% most_confused()

## -----------------------------------------------------------------------------
#  fls = list.files(paste(path,'images',sep = '/'),full.names = TRUE, recursive = TRUE)[c(250,500,700)]
#  fls
#  
#  #[1] "oxford-iiit-pet/images/american_bulldog_142.jpg"          "oxford-iiit-pet/images/american_pit_bull_terrier_188.jpg"
#  #[3] "oxford-iiit-pet/images/basset_hound_188.jpg"
#  
#  result = learn %>% predict(fls)
#  
#  str(result)

