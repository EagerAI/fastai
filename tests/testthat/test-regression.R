

context("regression")

source("utils.R")

test_succeeds('iris load', {
  df = iris
  df$Species = as.numeric(as.factor(df$Species))
})

test_succeeds('dls create', {
  procs = list(FillMissing(),Categorify(),Normalize())
  dls = TabularDataTable(df, procs, NULL, names(iris)[1:4],
                         y_names="Species", splits = list(c(1:120),c(121:150))) %>%
    dataloaders(bs=10)
})


test_succeeds('tabular ops create model', {
  model = dls %>% tabular_learner(layers=c(200,100), metrics=list(rmse(),mse()))
})

test_succeeds('tabular ops dims==batch', {
  dls %>% one_batch(TRUE) -> list_1
  # no embeddings
  expect_equal(dim(list_1[[1]]), c(10,0))
  expect_equal(dim(list_1[[2]]), c(10,4))
  expect_equal(dim(list_1[[3]]), c(10,1))
})

test_succeeds('tabular ops train model', {
  model %>% fit(1,1e-2)
})

test_succeeds('tabular ops predict', {
  res = model %>% predict(df[4,])
  expect_equal(names(res),colnames(iris)[5])
  expect_length(res,1)
})










