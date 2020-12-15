
#' @title ShapInterpretation
#'
#' @description Base interpereter to use the `SHAP` interpretation library
#'
#'
#' @param learn learner/model
#' @param test_data should be either a Pandas dataframe or a TabularDataLoader. If not, 100 random rows of
#' the training data will be used instead.
#' @param link link can either be "identity" or "logit". A generalized linear model link to connect
#' the feature importance values to the model output. Since the feature importance values, phi, sum up
#' to the model output, it often makes sense to connect them to the ouput with a link function where
#' link(outout) = sum(phi). If the model output is a probability then the LogitLink link function makes
#' the feature importance values have log-odds units.
#' @param l1_reg can be an integer value representing the number of features, "auto", "aic", "bic", or
#' a float value. The l1 regularization to use for feature selection (the estimation procedure is based
#' on a debiased lasso). The auto option currently uses "aic" when less that 20% of the possible sample
#' space is enumerated, otherwise it uses no regularization.
#' @param n_samples can either be "auto" or an integer value. This is the number of times to re-evaluate
#' the model when explaining each predictions. More samples leads to lower variance estimations of the SHAP values
#' @return None
#' @export
ShapInterpretation <- function(learn, test_data = NULL, link = "identity", l1_reg = "auto", n_samples = 128) {

  args <- list(
    learn = learn,
    test_data = test_data,
    link = link,
    l1_reg = l1_reg,
    n_samples = as.integer(n_samples)
  )

  do.call(fastinf()$tabular$ShapInterpretation, args)

}



#' @title Decision_plot
#'
#' @description Visualizes a model's decisions using cumulative SHAP values.
#'
#' @param object ShapInterpretation object
#' @param class_id is used to indicate the class of interest for a classification model.
#' It can either be an int or str representation for a class of choice. Each colored line in
#' the plot represents the model's prediction for a single observation.
#'
#' @param row_idx If no index is passed in to use from the data, it will default to the first ten samples
#' on the test set. Note:plotting too many samples at once can make the plot illegible.
#' @param dpi dots per inch
#' @param ... additional arguments
#' @return None
#' @export
decision_plot = function(object, class_id = 0, row_idx = -1, dpi = 200, ...) {
  args = list(
    class_id = as.integer(class_id),
    row_idx = as.integer(row_idx)
  )

  fastai2$vision$all$plt$close()
  do.call(object$decision_plot, args)
  #fastai2$vision$all$plt$tight_layout()
  tmp_d = gsub(tempdir(), replacement = '/', pattern = '\\', fixed = TRUE)
  #fastai2$vision$all$plt$rcParams$update(list('font.size' = 8L))
  fig = fastai2$vision$all$plt$gcf()
  fig$set_size_inches(18,10, forward=TRUE)
  fastai2$tabular$all$plt$savefig(paste(tmp_d, 'test.png', sep = '/'), dpi = as.integer(dpi), ...)

  img <- png::readPNG(paste(tmp_d, 'test.png', sep = '/'))
  if(interactive()) {
    try(dev.off(),TRUE)
  }
  grid::grid.raster(img)
  fastai2$vision$all$plt$close()

}





#' @title Dependence_plot
#'
#' @description Plots the value of a variable on the x-axis and the SHAP value of the same
#' variable on the y-axis. Accepts a class_id and variable_name.
#'
#' @param object ShapInterpretation object
#' @param class_id is used to indicate the class of interest for a classification model.
#' It can either be an int or str representation for a class of choice. This plot shows how the
#' model depends on the given variable. Vertical dispersion of the datapoints represent
#' interaction effects. Gray ticks along the y-axis are datapoints where the variable's values were NaN.
#'
#' @param variable_name the name of the column
#' @param dpi dots per inch
#' @param ... additional arguments
#' @return None
#' @export
dependence_plot = function(object, variable_name = "", class_id = 0, dpi = 200, ...) {
  args = list(
    class_id = as.integer(class_id),
    variable_name = variable_name
  )

  fastai2$vision$all$plt$close()
  do.call(object$dependence_plot, args)
  #fastai2$vision$all$plt$tight_layout()
  tmp_d = gsub(tempdir(), replacement = '/', pattern = '\\', fixed = TRUE)
  fig = fastai2$vision$all$plt$gcf()
  fig$set_size_inches(18,10, forward=TRUE)
  fastai2$tabular$all$plt$savefig(paste(tmp_d, 'test.png', sep = '/'), dpi = as.integer(dpi), ...)

  img <- png::readPNG(paste(tmp_d, 'test.png', sep = '/'))
  if(interactive()) {
    try(dev.off(),TRUE)
  }
  grid::grid.raster(img)
  fastai2$vision$all$plt$close()

}







#' @title Summary_plot
#'
#' @description Displays the SHAP values (which can be interpreted for feature importance)
#'
#' @param object ShapInterpretation object
#' @param dpi dots per inch
#' @param ... additional arguments
#' @return None
#' @export
summary_plot = function(object, dpi = 200, ...) {
  fastai2$vision$all$plt$close()
  object$summary_plot()

  #fastai2$vision$all$plt$tight_layout()
  tmp_d = gsub(tempdir(), replacement = '/', pattern = '\\', fixed = TRUE)
  fig = fastai2$vision$all$plt$gcf()
  fig$set_size_inches(18,10, forward=TRUE)
  fastai2$tabular$all$plt$savefig(paste(tmp_d, 'test.png', sep = '/'), dpi = as.integer(dpi), ...)

  img <- png::readPNG(paste(tmp_d, 'test.png', sep = '/'))
  if(interactive()) {
    try(dev.off(),TRUE)
  }
  grid::grid.raster(img)
  fastai2$vision$all$plt$close()

}





#' @title Waterfall_plot
#'
#' @description Plots an explanation of a single prediction as a waterfall plot. Accepts a row_index and class_id.
#'
#' @param object ShapInterpretation object
#' @param row_idx is the index of the row chosen in test_data to be analyzed, which defaults to zero.
#' @param class_id Accepts a class_id which is used to indicate the class of interest for a
#' classification model. It can either be an int or str representation for a class of choice.
#' @param dpi dots per inch
#' @param ... additional arguments
#' @return None
#' @export
waterfall_plot = function(object, row_idx = NULL, class_id = 0, dpi = 200, ...) {

  args = list(
    row_idx = row_idx,
    class_id = as.integer(class_id)
  )
  fastai2$vision$all$plt$close()
  if(!is.null(args[['row_idx']]))
    args[['row_idx']] = as.integer(args[['row_idx']])

  do.call(object$waterfall_plot, args)

  #fastai2$vision$all$plt$figure(figsize=c(8L, 6L), dpi=120)
  #fastai2$vision$all$plt$tight_layout()
  tmp_d = gsub(tempdir(), replacement = '/', pattern = '\\', fixed = TRUE)

  fig = fastai2$vision$all$plt$gcf()
  fig$set_size_inches(18,10, forward=TRUE)
  fastai2$tabular$all$plt$savefig(paste(tmp_d, 'test.png', sep = '/'), dpi = as.integer(dpi),bbox_inches="tight", ...)

  img <- png::readPNG(paste(tmp_d, 'test.png', sep = '/'))
  if(interactive()) {
    try(dev.off(),TRUE)
  }
  grid::grid.raster(img)
  fastai2$vision$all$plt$close()

}





#' @title Force_plot
#'
#' @description Visualizes the SHAP values with an added force layout. Accepts a class_id which
#' is used to indicate the class of interest for a classification model.
#'
#' @param object ShapInterpretation object
#' @param class_id Accepts a class_id which is used to indicate the class of interest for a
#' classification model. It can either be an int or str representation for a class of choice.
#' @param ... additional arguments
#' @return None
#' @export
force_plot = function(object, class_id = 0, ...) {
  fastai2$vision$all$plt$close()
  tempDir <- tempfile()
  dir.create(tempDir)
  shap()$save_html(paste(tempDir,'test.html',sep = '/'),
                 object$force_plot(class_id = as.integer(class_id)), ...)
  htmlFile <- file.path(tempDir, "test.html")
  viewer <- getOption("viewer")
  viewer(htmlFile)

}



