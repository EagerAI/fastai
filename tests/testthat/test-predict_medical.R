context("Medical_predict")

source("utils.R")

test_succeeds('download URLs_SIIM_SMALL', {
  if(!dir.exists('siim_small')) {
    URLs_SIIM_SMALL()
  }
})

test_succeeds('prepare dataloader and model', {
  items = get_dicom_files("siim_small/train/")
  df = data.table::fread("siim_small/labels.csv")

  pneumothorax = DataBlock(blocks = list(ImageBlock(cls = Dicom()), CategoryBlock()),
                           get_x = function(x) {paste('siim_small', x[[1]], sep = '/')},
                           get_y = function(x) {paste(x[[2]])},
                           batch_tfms = aug_transforms(size = 224))

  dls = pneumothorax %>% dataloaders(as.matrix(df))
  learn = cnn_learner(dls, resnet34(), metrics = accuracy)
})


test_succeeds('predict medical', {
  result = learn %>% predict(as.character(items[0]))
})

test_succeeds('predict medical list length', {
  expect_length(result, 2)
})

test_succeeds('predict medical list probability list length is 2', {
  expect_length(result[[1]],2)
})

test_succeeds('predict medical list label list length is 1', {
  expect_length(result[[2]],1)
})

test_succeeds('predict medical list probability names are right', {
  expect_equal(names(result[[1]]), dls$vocab$items$items)
})




