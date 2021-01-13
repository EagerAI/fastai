


context("Audio")

source("utils.R")

test_succeeds('download SPEAKERS10', {
  if(!dir.exists('SPEAKERS10')) {
    URLs_SPEAKERS10()
  }
  path_dig = 'SPEAKERS10'
})

test_succeeds('test audio extensions', {
  #expect_equal(audio_extensions()[1:6],c(".aif",".aifc",".aiff",".au",".m3u",".mp2"))
  #expect_length(audio_extensions(),35)
})

test_succeeds('read sample audio and plot', {
  fnames = get_files(path_dig, extensions = audio_extensions())
  at = AudioTensor_create(fnames[0])
  at %>% show() %>% plot(dpi = 200)
})


test_succeeds('audio Voice freq max and sample rate', {
  cfg = Voice()
  expect_equal(c(cfg$f_max, cfg$sample_rate), c(8e3,16e3))
})

test_succeeds('audio spec from cfg and signal resize', {
  aud2spec = AudioToSpec_from_cfg(cfg)
  crop1s = ResizeSignal(1000)
})

test_succeeds('audio dataloader', {
  item_tfms = list(ResizeSignal(1000), aud2spec)

  get_y = function(x) substring(x$name[1],1,1)

  aud_digit = DataBlock(blocks = list(AudioBlock(), CategoryBlock()),
                        get_items = get_audio_files,
                        splitter = RandomSplitter(),
                        item_tfms = item_tfms,
                        get_y = get_y)

  dls = aud_digit %>% dataloaders(source = path_dig, bs = 64)

  dls %>% show_batch(figsize = c(15, 8.5), nrows = 3, ncols = 3, max_n = 9, dpi = 180)
})


test_succeeds('audio learner with 1 input channel', {
  learn = cnn_learner(dls,
                      resnet34(),
                      config = list("n_in" = 1L),
                      opt_func = Adam(),
                      metrics = accuracy())
})





