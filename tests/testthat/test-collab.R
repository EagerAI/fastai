

context("collab")

source("utils.R")

test_succeeds('download movie lens data', {
  if(!dir.exists('ml-100k')) {
    URLs_MOVIE_LENS_ML_100k()
  }
  user = 'userId'
  item = 'movieId'
  title = 'title'
})


#test_succeeds('read movie lens datas', {
#  ratings = fread('ml-100k/u.data', col.names = c(user,item,'rating','timestamp'))
#  movies = fread('ml-100k/u.item', col.names = c(item, 'title', 'date', 'N', 'url',
#                                                 paste('g',1:19,sep = '')))
#  rating_movie = ratings[movies[, .SD, .SDcols=c(item,title)], on = item]
#})

#test_succeeds('movie lens prepare dls', {
#  dls = CollabDataLoaders_from_df(rating_movie, seed=42, valid_pct=0.1, bs=64, item_name=title, path='ml-100k')
#})

#test_succeeds('movie lens data model fit', {
#  learn = collab_learner(dls, n_factors = 40, y_range=c(0, 5.5))
#  learn %>% fit_one_cycle(1, 5e-3,  wd = 1e-1)
#})



#test_succeeds('top movies bias/weights', {
#  top_movies = head(unique(rating_movie[ , count := .N, by = .(title)]
#                           [order(count,decreasing = T)]
#                           [, c('title','count')]),
#                    1e3)[['title']]
#  mean_ratings = unique(rating_movie[ , .(mean = mean(rating)), by = title])
#
#  movie_bias = learn %>% get_bias(top_movies, is_item = TRUE)
#
#  result = data.table(bias = movie_bias,
#                      title = top_movies)
#
#  res = merge(result, mean_ratings, all.y = FALSE)
#
#  res[order(bias, decreasing = TRUE)]
#
#  movie_w = learn %>% get_weights(top_movies, is_item = TRUE, convert = TRUE)
#})









