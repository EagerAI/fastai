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

## -----------------------------------------------------------------------------
#  encoder = create_body(resnet34(), pretrained = TRUE)
#  
#  arch = RetinaNet(encoder, get_c(dls), final_bias=-4)
#  
#  ratios = c(1/2,1,2)
#  scales = c(1,2**(-1/3), 2**(-2/3))
#  
#  crit = RetinaNetFocalLoss(scales = scales, ratios = ratios)
#  
#  nn = nn()
#  
#  retinanet_split = function(m) {
#    L(m$encoder,nn$Sequential(m$c5top6, m$p6top7, m$merges,
#                         m$smoothers, m$classifier, m$box_regressor))$map(params())
#  }

## -----------------------------------------------------------------------------
#  learn = Learner(dls, arch, loss_func = crit, splitter = retinanet_split)
#  
#  learn$freeze()
#  
#  learn %>% fit_one_cycle(10, slice(1e-5, 1e-4))

