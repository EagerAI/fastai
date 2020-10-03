## ----setup, include=FALSE-----------------------------------------------------
knitr::opts_chunk$set(echo = TRUE,eval = FALSE,echo = T)

## -----------------------------------------------------------------------------
#  # cat categories: https://www.purina.com/cats/cat-breeds
#  f_n = 'cats'
#  
#  if(!dir.exists(f_n)) {
#    dir.create(f_n)
#  }

## -----------------------------------------------------------------------------
#  library(rvest)
#  
#  download_pet = function(name, dest) {
#    query = name
#    query = gsub('\\s', '%20', query)
#  
#    search <- read_html(paste("https://www.google.com/search?site=&tbm=isch&q", query, sep = "="))
#  
#    urls <- search %>% html_nodes("img") %>% html_attr("src") %>% .[-1]
#  
#    fixed_name = gsub('\\s|[[:punct:]]', '_', name)
#  
#    for (i in 1:length(urls)) {
#      download.file(urls[i], destfile =
#    file.path(dest,
#        paste(
#          paste(fixed_name,
#                round(runif(1)*10000),
#                sep = '_'),
#          '.jpg', sep = ''
#        )
#      ), mode = 'wb'
#      )
#    }
#  }

## -----------------------------------------------------------------------------
#  cat_names = c('Balinese-Javanese Cat Breed', 'Chartreux Cat Breed',
#                'Norwegian Forest Cat Breed', 'Turkish Angora Cat Breed')

## -----------------------------------------------------------------------------
#  for (i in 1:length(cat_names)) {
#    download_pet(cat_names[i], f_n)
#    print(paste('Done',cat_names[i]))
#  }

## -----------------------------------------------------------------------------
#  library(fastai)
#  library(magrittr)
#  
#  path = 'cats'
#  fnames = get_image_files(path)
#  
#  fnames[1]
#  # cats/Turkish_Angora_Cat_Breed_8583.jpg

## -----------------------------------------------------------------------------
#  dls = ImageDataLoaders_from_name_re(
#    path, fnames, pat='(.+)_\\d+.jpg$',
#    item_tfms  = Resize(size = 200), bs = 15,
#    batch_tfms = list(aug_transforms(size = 224, min_scale = 0.75),
#                    Normalize_from_stats( imagenet_stats() )
#    ),
#    device = 'cuda'
#  )
#  
#  
#  dls %>% show_batch(dpi = 200)

## -----------------------------------------------------------------------------
#  learn = cnn_learner(dls, resnet50(), metrics = list(accuracy, error_rate))
#  
#  learn$recorder$train_metrics = TRUE
#  

## -----------------------------------------------------------------------------
#  learn %>% fit_one_cycle(n_epoch = 5, 1e-3)

## -----------------------------------------------------------------------------
#  fnames[1]
#  
#  # cats/Turkish_Angora_Cat_Breed_8583.jpg
#  
#  learn %>% predict(as.character(fnames[1]))

