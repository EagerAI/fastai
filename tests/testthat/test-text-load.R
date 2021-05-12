

context("text dls")

source("utils.R")


test_succeeds('read movie data', {
  df = text2vec::movie_review[1:100,]
  df$sentiment<- as.factor(df$sentiment)
})

test_succeeds('text dataloader', {
  data_block = DataBlock(
    blocks = list(TextBlock_from_df(text_cols="review"), CategoryBlock()),
    get_x = ColReader("text"),
    get_y = ColReader('sentiment'),
    splitter=RandomSplitter(0.1))

  data_iterator = data_block %>% dataloaders(source=df, bs=4)
  data_iterator %>% one_batch()
})

test_succeeds('predict text with AWS', {
  learnR <- language_model_learner(data_iterator, AWD_LSTM(), drop_mult = 0.3, metrics = list(accuracy, Perplexity()))
  learnR %>% predict(df$review[1])
})




