

context("Medical_image_rendering")

source("utils.R")

test_succeeds('download dcm file GITHUB', {
  download.file('https://github.com/henry090/fastai/raw/master/files/hemorrhage.dcm',destfile = 'hemorrhage.dcm')
})


test_succeeds('read/subplot dcm file GITHUB', {
  img = dcmread('hemorrhage.dcm')
})

test_succeeds('render dcm file GITHUB', {
  dicom_windows = dicom_windows()
  scale = list(FALSE, TRUE, dicom_windows$brain, dicom_windows$subdural)
  titles = c('raw','normalized','brain windowed','subdural windowed')

  one = subplots()
  fig = one[[1]]
  axs = one[[2]]

  for (i in 1:4) {
    img %>% show(scale = scale[[i]],
                 ax = axs[[i]],
                 title=titles[i])
  }

  img %>% plot(dpi = 250)
})

test_succeeds('cmap for dcm file GITHUB', {
  img %>% show(cmap = cm()$gist_ncar, figsize = c(6,6))
  img %>% plot()
})

#test_succeeds('complex ggplot for dcm file GITHUB', {
#  types = c('raw', 'normalized', 'brain', 'subdural')
#  p_ = list()
#  for ( i in 1:length(types)) {
#    p = nandb::matrix_raster_plot(img %>% get_dcm_matrix(type = types[i]))
#    p_[[i]] = p
#  }
#
#  ggpubr::ggarrange(p_[[1]], p_[[2]], p_[[3]], p_[[4]], labels = types)
#})

#test_succeeds('ggplot for dcm file GITHUB', {
#  res = img %>% mask_from_blur(win_brain()) %>%
#    mask2bbox()
#
#  types = c('raw', 'normalized', 'brain', 'subdural')
#
#   #colors for matrix filling
#  colors = list(viridis::inferno(30), viridis::magma(30),
#                viridis::plasma(30), viridis::cividis(30))
#  scan_ = c('uniform_blur2d', 'gauss_blur2d')
#  p_ = list()
#
#  for ( i in 1:length(types)) {
#    if(i == 3) {
#      scan = scan_[1]
#    } else if (i==4) {
#      scan = scan_[2]
#    } else {
#      scan = ''
#    }
#
#     #crop with x/y_lim functions from ggplot
#    if(i==2) {
#      p = nandb::matrix_raster_plot(img %>% get_dcm_matrix(type = types[i],
#                                                           scan = scan),
#                                    colours = colors[[i]])
#      p = p + ylim(c(res[[1]][[1]],res[[2]][[1]])) + xlim(c(res[[1]][[2]],res[[2]][[2]]))
#
#       # zoom image (25 %)
#    } else if (i==4) {
#
#      img2 = img
#      #img2 %>% zoom(0.25)
#      p = nandb::matrix_raster_plot(img2 %>% get_dcm_matrix(type = types[i],
#                                                            scan = scan),
#                                    colours = colors[[i]])
#    } else {
#      p = nandb::matrix_raster_plot(img %>% get_dcm_matrix(type = types[i],
#                                                           scan = scan),
#                                    colours = colors[[i]])
#    }
#
#    p_[[i]] = p
#  }
#
#  ggpubr::ggarrange(p_[[1]],
#                    p_[[2]],
#                    p_[[3]],
#                    p_[[4]],
#                    labels = paste(types[1:4],
#                                   paste(c('','',scan_))[1:4])
#  )
#})









