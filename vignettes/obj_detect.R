## ----setup, include=FALSE-----------------------------------------------------
knitr::opts_chunk$set(echo = TRUE,eval = FALSE,echo = T)

## -----------------------------------------------------------------------------
#  URLs_COCO_TINY()

## -----------------------------------------------------------------------------
#  c(images, lbl_bbox) %<-% get_annotations('coco_tiny/train.json')
#  
#  names(lbl_bbox) = images
#  
#  img2bbox = lbl_bbox

## -----------------------------------------------------------------------------
#  get_y = list(function(o) img2bbox[[o$name]][[1]],
#               function(o) as.list(img2bbox[[o$name]][[2]]))
#  
#  coco = DataBlock(blocks = list(ImageBlock(), BBoxBlock(), BBoxLblBlock()),
#                   get_items = get_image_files(),
#                   splitter = RandomSplitter(),
#                   get_y = get_y,
#                   item_tfms = Resize(128),
#                   batch_tfms = aug_transforms(),
#                   n_inp = 1)
#  
#  dls = coco %>% dataloaders('coco_tiny/train')
#  dls %>% show_batch(max_n = 12)

