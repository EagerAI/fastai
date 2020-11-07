

context("collab")

source("utils.R")


test_succeeds('read movie lens datas', {
  rating_movie = fread('https://raw.githubusercontent.com/henry090/fastai/master/files/rating_movie.csv')
})

test_succeeds('movie lens prepare dls', {
  dls = CollabDataLoaders_from_df(rating_movie, seed=42, valid_pct=0.1, bs=64, item_name='title')
})

test_succeeds('movie lens data model fit', {
  learn = collab_learner(dls, n_factors = 40, y_range=c(0, 5.5))
  learn %>% fit_one_cycle(1, 5e-3,  wd = 1e-1)
})



test_succeeds('top movies bias/weights', {
  top_movies = unique(rating_movie$title)

  movie_bias = learn %>% get_bias(top_movies, is_item = TRUE)

  result = data.table(bias = movie_bias,
                      title = top_movies)

  movie_w = learn %>% get_weights(top_movies, is_item = TRUE, convert = TRUE)
})









