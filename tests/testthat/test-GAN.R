

context("GAN")

source("utils.R")

test_succeeds('download URLs_HORSE_2_ZEBRA', {
  if(!dir.exists('horse2zebra')) {
    URLs_HORSE_2_ZEBRA()
  }
})


test_succeeds('dataloader URLs_HORSE_2_ZEBRA', {
  horse2zebra = 'horse2zebra'
  trainA_path = file.path(horse2zebra,'trainA')
  trainB_path = file.path(horse2zebra,'trainB')
  testA_path = file.path(horse2zebra,'testA')
  testB_path = file.path(horse2zebra,'testB')

  if(reticulate::py_module_available('upit')) {
    dls = get_dls(trainA_path, trainB_path, num_A = 130,load_size = 270,crop_size = 144,bs=4)
  }
})

test_succeeds('CycleGAN model', {
  if(reticulate::py_module_available('upit')) {
    cycle_gan = CycleGAN(3,3,64)
    learn = cycle_learner(dls, cycle_gan)
  }
})


test_succeeds('CycleGAN model predict', {
  if(reticulate::py_module_available('upit')) {
    learn %>% get_preds_cyclegan(testA_path, './h2z-preds')
    expect_true(dir.exists('h2z-preds'))
    learn %>% show_results()
  }
})

test_succeeds('CycleGAN model predict folder exists', {
  if(reticulate::py_module_available('upit')) {
    expect_true(dir.exists('h2z-preds'))
    learn %>% show_results()
  }
})



