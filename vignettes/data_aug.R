## ----setup, include=FALSE-----------------------------------------------------
knitr::opts_chunk$set(echo = TRUE,eval = FALSE,echo = T)

## -----------------------------------------------------------------------------
#  img = fastai::Image_create('files/cat.jpeg')

## -----------------------------------------------------------------------------
#  img %>% show() %>% plot()

## -----------------------------------------------------------------------------
#  img_res = list(img, img$flip_lr())
#  titles = c('original', 'flipped')
#  
#  c(fig, axs) %<-% subplots(1,2)
#  
#  for (i in 1:2) {
#    img_res[[i]] %>% show_image(ax = axs[[i]],
#                 title=titles[i])
#  }
#  
#  img %>% plot(dpi = 250)

## -----------------------------------------------------------------------------
#  c(fig, axs) %<-% subplots(2, 4)
#  
#  for (i in 1:8) {
#    show_image(DihedralItem(p = 1.)(img, split_idx = 0), ctx = axs[[i]])
#  }
#  
#  img %>% plot(dpi = 250)

## -----------------------------------------------------------------------------
#  sz = c(300L, 500L, 700L)
#  size = paste('Size', sz)
#  
#  c(fig, axs) %<-% subplots(1, 3, figsize = c(12, 4))
#  
#  for (i in 1:3) {
#    show_image(img$crop_pad(sz[i]), ctx = axs[[i]], title = size[i])
#  }
#  
#  img %>% plot(dpi = 250)

## -----------------------------------------------------------------------------
#  pad_modes = c('border', 'reflection', 'zeros')
#  
#  c(fig, axs) %<-% subplots(1, 3, figsize = c(12, 4))
#  
#  for (i in 1:3) {
#    show_image(img$crop_pad(c(600L,700L), pad_mode = pad_modes[i]),
#               ctx = axs[[i]], title = pad_modes[i])
#  }
#  
#  img %>% plot(dpi = 250)

## -----------------------------------------------------------------------------
#  c(fig, axs) %<-% subplots(1, 3, figsize = c(12, 4))
#  
#  ff = RandomCrop(100)
#  
#  for (i in 1:3) {
#    show_image(ff(img), ctx = axs[[i]])
#  }
#  
#  img %>% plot(dpi = 250)

## -----------------------------------------------------------------------------
#  c(fig, axs) %<-% subplots(1, 3, figsize = c(12, 4))
#  
#  ff = RandomCrop(100L)
#  
#  for (i in 1:3) {
#    show_image(ff(img, split_idx = 1L), ctx = axs[[i]])
#  }
#  
#  img %>% plot(dpi = 250)

## -----------------------------------------------------------------------------
#  resize = c('squish', 'pad', 'crop')
#  
#  c(fig, axs) %<-% subplots(1, 3, figsize = c(12, 4))
#  
#  for (i in 1:3) {
#    rsz = Resize(256, method = resize[i])
#    show_image(rsz(img, split_idx = 0L), ctx = axs[[i]], title = resize[i])
#  }
#  
#  img %>% plot(dpi = 250)

## -----------------------------------------------------------------------------
#  c(fig, axs) %<-% subplots(3, 3, figsize = c(9, 9))
#  
#  ff = RandomResizedCrop(100)
#  
#  for (i in 1:9) {
#    show_image(ff(img), ctx = axs[[i]])
#  }
#  
#  img %>% plot(dpi = 250)

