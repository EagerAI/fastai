context("tabular")

source("utils.R")

test_succeeds('dataset load', {
  df = fread('https://github.com/henry090/fastai/raw/master/files/adult.csv')
  dep_var = 'salary'
  cat_names = c('workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race')
  cont_names = c('age', 'fnlwgt', 'education-num')
})

test_succeeds('tabular ops FillMissing', {
  expect_equal(capture.output(FillMissing()),"<class 'fastai.tabular.core.FillMissing'>")
})

test_succeeds('tabular ops Categorify', {
  expect_equal(capture.output(Categorify()),"<class 'fastai.tabular.core.Categorify'>")
})

test_succeeds('tabular ops Normalize', {
  expect_equal(capture.output(Normalize()),"<class 'fastai.data.transforms.Normalize'>")
})

test_succeeds('tabular ops dataloader', {
  procs = list(FillMissing(),Categorify(),Normalize())
  dls = TabularDataTable(df, procs, cat_names, cont_names,
                         y_names="salary", splits = list(c(1:32000),c(32001:32561))) %>%
    dataloaders(bs=10)
  expect_length(one_batch(dls, convert = FALSE), 3)
})


test_succeeds('tabular ops create model', {
  model = dls %>% tabular_learner(layers=c(200,100), metrics=accuracy)
})

test_succeeds('tabular ops train model', {
  model %>% fit(1,1e-2)
})

test_succeeds('tabular ops predict', {
  res = model %>% predict(df[4,])
  expect_length(res, 2)
})

test_succeeds('tabular ops get optimal lr', {
  model %>% lr_find()
  df = model %>% lr_find_()
  expect_true(is.data.frame(df))
})


test_succeeds('tabular ops confusion matrix', {
  conf = model %>% get_confusion_matrix() %>% as.data.frame()
  expect_equal(names(conf),rownames(conf))
  expect_equal(length(names(conf)),2)
  expect_equal(length(rownames(conf)),2)
})










