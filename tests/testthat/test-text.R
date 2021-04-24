context("text")

source("utils.R")

test_succeeds('download URLs_IMDB', {
  URLs_IMDB()
})


test_succeeds('download URLs_IMDB Datablock', {
  path = 'imdb'
  bs = 20
  imdb_lm = DataBlock(blocks=list(TextBlock_from_folder(path, is_lm = TRUE)),
                      get_items = partial(get_text_files(),
                                          folders = c('train', 'test', 'unsup')),
                      splitter = RandomSplitter(0.1))
})

test_succeeds('download URLs_IMDB Datablock dataloaders', {
  #dbunch_lm = imdb_lm %>% dataloaders(source = path, path = path, bs = bs, seq_len = 10,num_workers=0L)
})

test_succeeds('download URLs_IMDB Datablock leaner', {
  #learn = language_model_learner(dbunch_lm, AWD_LSTM(), drop_mult = 0.3,
  #                               metrics = list(accuracy(), Perplexity()))
})
