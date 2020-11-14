## ----setup, include=FALSE-----------------------------------------------------
knitr::opts_chunk$set(echo = TRUE,eval = FALSE,echo = T)

## -----------------------------------------------------------------------------
#  URLs_SIIM_SMALL()

## -----------------------------------------------------------------------------
#  library(fastai)
#  library(magrittr)
#  library(zeallot)
#  
#  items = get_dicom_files("siim_small/train/")
#  items
#  
#  c(trn,val) %<-% RandomSplitter()(items)
#  
#  patient = 7
#  xray_sample = dcmread(items[patient])
#  
#  xray_sample %>% show() %>% plot()

## -----------------------------------------------------------------------------
#  # gather data
#  items_list = items$items
#  
#  dicom_dataframe = data.frame()
#  
#  for(i in 1:length(items_list)) {
#    res = dcmread(as.character(items_list[[i]])) %>% to_matrix(matrix = FALSE)
#    dicom_dataframe = dicom_dataframe %>% rbind(res)
#    if(i %% 50 == 0) {
#      print(i)
#    }
#  }

## -----------------------------------------------------------------------------
#  df = data.table::fread("siim_small/labels.csv")
#  
#  pneumothorax = DataBlock(blocks = list(ImageBlock(cls = Dicom()), CategoryBlock()),
#                           get_x = function(x) {paste('siim_small', x[[1]], sep = '/')},
#                           get_y = function(x) {paste(x[[2]])},
#                           batch_tfms = aug_transforms(size = 224))
#  
#  dls = pneumothorax %>% dataloaders(as.matrix(df))
#  
#  dls %>% show_batch(max_n = 16)

## -----------------------------------------------------------------------------
#  learn = cnn_learner(dls, resnet34(), metrics = accuracy)
#  summary(learn)

## -----------------------------------------------------------------------------
#  learn %>% fit_one_cycle(3)

## -----------------------------------------------------------------------------
#  learn %>% show_results()

