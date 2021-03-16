

context("Medical")

source("utils.R")

test_succeeds('download URLs_SIIM_SMALL', {
  if(!dir.exists('siim_small')) {
    URLs_SIIM_SMALL()
  }
})


test_succeeds('read dcm URLs_SIIM_SMALL', {
  items = get_dicom_files("siim_small/train/")

  gn = RandomSplitter()(items)
  trn = gn[[1]]
  val = gn[[2]]

  patient = 7
  xray_sample = dcmread(items[patient])

  xray_sample %>% show() %>% plot()
})

test_succeeds('gather dcm URLs_SIIM_SMALL', {
  # gather data
  items_list = items$items

  dicom_dataframe = data.frame()

  for(i in 1:length(items_list)) {
    res = dcmread(as.character(items_list[[i]])) %>% to_matrix(matrix = FALSE)
    dicom_dataframe = dicom_dataframe %>% rbind(res)
    if(i %% 50 == 0) {
      print(i)
    }
  }

  #expect_length(tibble::tibble(head(dicom_dataframe,6)),42)
  #expect_equal(ncol(tibble::tibble(head(dicom_dataframe,6))),42)

})

test_succeeds('datalaoder and block for URLs_SIIM_SMALL', {
  df = data.table::fread("siim_small/labels.csv")

  pneumothorax = DataBlock(blocks = list(ImageBlock(cls = Dicom()), CategoryBlock()),
                           get_x = function(x) {paste('siim_small', x[[1]], sep = '/')},
                           get_y = function(x) {paste(x[[2]])},
                           batch_tfms = list(aug_transforms(size = 224),
                                             Normalize_from_stats( imagenet_stats() )
                           ))

  dls = pneumothorax %>% dataloaders(as.matrix(df))

  dls %>% show_batch(max_n = 16)
})






