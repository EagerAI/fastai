context("tabular")

source("utils.R")

test_succeeds('dataset load', {
  # download
  URLs_ADULT_SAMPLE()

  df = data.table::fread('adult_sample/adult.csv')
  df = df[1:2561,]
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
                         y_names="salary", splits = list(c(1:2000),c(2001:2561))) %>%
    dataloaders(bs=10)
  expect_length(one_batch(dls, convert = FALSE), 3)
})


test_succeeds('tabular ops create model', {
  model = dls %>% tabular_learner(layers=c(200,100), metrics=accuracy)
})

test_succeeds('tabular ops dims==batch', {
  dls %>% one_batch(TRUE) -> list_1
  expect_equal(dim(list_1[[1]]), c(10,7))
  expect_equal(dim(list_1[[2]]), c(10,3))
  expect_equal(dim(list_1[[3]]), c(10,1))
})

test_succeeds('tabular ops bs find', {
  #bss = model %>% bs_find(1e-3)
  #model %>% plot_bs_find()
  #expect_s3_class(bss, 'data.frame')
})


test_succeeds('tabular ops train model', {
  model %>% fit(1,1e-2)
})

test_succeeds('tabular ops predict', {
  res = model %>% predict(df[4,])
  expect_length(res, 3)
})

test_succeeds('tabular ops get optimal lr', {
  df = model %>% lr_find()
  model %>% plot_lr_find()
  expect_true(is.data.frame(df))
})


test_succeeds('tabular ops confusion matrix', {
  conf = model %>% get_confusion_matrix() %>% as.data.frame()
  expect_equal(names(conf),rownames(conf))
  expect_equal(length(names(conf)),2)
  expect_equal(length(rownames(conf)),2)
})

test_succeeds('tabular ops confusion matrix via class-n interp', {
  interp = ClassificationInterpretation_from_learner(model)
  interp %>% plot_confusion_matrix(dpi = 90, figsize = c(6,6))
})


test_succeeds('tabular ops shap intep object', {
  exp = ShapInterpretation(model,n_samples = 6)
})


test_succeeds('tabular ops shap decision plot', {
  exp %>% decision_plot(class_id = 1, row_idx = 2)
})


test_succeeds('tabular ops shap dependence plot', {
  exp %>% dependence_plot('age', class_id = 0)
})


test_succeeds('tabular ops shap summary plot', {
  exp %>% summary_plot()
})



test_succeeds('tabular ops shap waterfall plot', {
  exp %>% waterfall_plot(row_idx=2)
})


test_succeeds('tabular ops shap JS plot', {
  #exp %>% force_plot(class_id = 0)
})





