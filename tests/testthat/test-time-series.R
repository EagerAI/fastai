

#context("regression")

#source("utils.R")

#test_succeeds('iris load', {
#  df = iris
#  df$Species = as.numeric(as.factor(df$Species))
#  df_train = df[1:120,]
#  df_test = df[121:150,]
#})

#test_succeeds('dls create', {
#  x_cols = names(iris)[1:4]
#  dls = TSDataLoaders_from_dfs(df_train, df_test, x_cols = x_cols, label_col = 'Species', bs=60,
#                               y_block = RegressionBlock())
#})


#test_succeeds('inception model', {
#  inception = create_inception(1, 1)
#  learn = Learner(dls, inception, metrics=list(mae(), rmse()))
#})

#test_succeeds('inception model train', {
# learn %>% fit_one_cycle(10, 1e-5, cbs = EarlyStoppingCallback(patience = 5))
#})

#test_succeeds('inception model predict', {
#  result = learn %>% predict(df_test)
#  expect_equal(names(result),names(iris)[5])
#})








