




#' @title TabularDataTable
#'
#' @description A `Tabular` object with transforms
#'
#'
#' @param df A DataFrame of your data
#' @param procs procs
#' @param cat_names the names of the categorical variables
#' @param cont_names the names of the continuous variables
#' @param y_names the names of the dependent variables
#' @param y_block the TransformBlock to use for the target
#' @param splits How to split your data
#' @param do_setup A parameter for if Tabular will run the data through the procs upon initialization
#' @param device cuda or cpu
#' @param inplace If True, Tabular will not keep a separate copy of your original DataFrame in memory
#' @param reduce_memory fastai will attempt to reduce the overall memory usage
#'
#' @export
TabularDataTable <- function(df, procs = NULL, cat_names = NULL, cont_names = NULL,
                             y_names = NULL, y_block = NULL, splits = NULL,
                             do_setup = TRUE, device = NULL, inplace = FALSE, reduce_memory = TRUE) {

  args <- list(
    df = df,
    procs = procs,
    cat_names = cat_names,
    cont_names = cont_names,
    y_names = y_names,
    y_block = y_block,
    splits = splits,
    do_setup = do_setup,
    device = device,
    inplace = inplace,
    reduce_memory = reduce_memory
  )

  if(!is.null(splits))
    args$splits = list(as.integer(splits[[1]]-1),as.integer(splits[[2]]-1))

  do.call(tabular$TabularPandas, args)

}


#' @title Trainable_params
#'
#' @description Return all trainable parameters of `m`
#'
#'
#' @param m m
#'
#' @export
trainable_params <- function(m) {

  if(missing(m)) {
    tabular$trainable_params
  } else {
    tabular$trainable_params(
      m = m
    )
  }

}

#' @title Tabular_learner
#'
#' @description Get a `Learner` using `dls`, with `metrics`, including a `TabularModel` created using the remaining params.
#'
#'
#' @param dls It is a DataLoaders object.
#' @param layers layers
#' @param emb_szs emb_szs
#' @param config config
#' @param n_out n_out
#' @param y_range y_range
#' @param loss_func It can be any loss function you like.
#' @param opt_func It will be used to create an optimizer when Learner.fit is called.
#' @param lr It is learning rate.
#' @param splitter It is a function that takes self.model and returns a list of parameter groups (or just one parameter group if there are no different parameter groups)
#' @param cbs It is one or a list of Callbacks to pass to the Learner.
#' @param metrics It is an optional list of metrics, that can be either functions or Metrics.
#' @param path İt is used to save and/or load models.Often path will be inferred from dls, but you can override it or pass a Path object to model_dir. Make sure you can write in path/model_dir!
#' @param model_dir İt is used to save and/or load models.Often path will be inferred from dls, but you can override it or pass a Path object to model_dir. Make sure you can write in path/model_dir!
#' @param wd It is the default weight decay used when training the model.
#' @param wd_bn_bias It controls if weight decay is applied to BatchNorm layers and bias.
#' @param train_bn It controls if BatchNorm layers are trained even when they are supposed to be frozen according to the splitter.
#' @param moms The default momentums used in Learner.fit_one_cycle.
#'
#' @export
tabular_learner <- function(dls, layers = NULL, emb_szs = NULL, config = NULL,
                            n_out = NULL, y_range = NULL, loss_func = NULL,
                            opt_func = Adam(), lr = 0.001,
                            splitter = trainable_params(), cbs = NULL,
                            metrics = NULL, path = NULL, model_dir = "models",
                            wd = NULL, wd_bn_bias = FALSE, train_bn = TRUE,
                            moms = list(0.95, 0.85, 0.95)) {

 args <- list(
    dls = dls,
    layers = layers,
    emb_szs = emb_szs,
    config = config,
    n_out = n_out,
    y_range = y_range,
    loss_func = loss_func,
    opt_func = opt_func,
    lr = lr,
    splitter = splitter,
    cbs = cbs,
    metrics = metrics,
    path = path,
    model_dir = model_dir,
    wd = wd,
    wd_bn_bias = wd_bn_bias,
    train_bn = train_bn,
    moms = moms
  )

 if(is.list(layers)) {
   args$layers <- as.list(
     as.integer(
       unlist(args$layers)
     )
   )
 } else{
   args$layers <- as.integer(args$layers)
 }

 do.call(tabular$tabular_learner, args)

}

#' @title Fit
#' @description Fit the model on this learner with `lr` learning rate, `wd` weight decay for `epochs` with `callbacks`.
#'
#' @param epochs epochs
#' @param lr lr
#' @param wd wd
#' @param callbacks callbacks
#'
#' @export
fit.fastai.tabular.learner.TabularLearner <- function(object, ...) {

  args <- list(
    ...
  )

  if(!is.null(args[[1]]) & is.null(names(args[[1]]))) {
    args[[1]] = as.integer(args[[1]])
  }

  find_epoch = which(names(args)=='n_epoch')

  if(length(find_epoch)>0) {
    args[[find_epoch]] = as.integer(args[[find_epoch]])
  }

  # fit the model
  do.call(object$fit, args)

  if (length(length(object$recorder$values))==1) {
    history = data.frame(values = do.call(rbind,lapply(1:length(object$recorder$values),
                                                         function(x) object$recorder$values[[x]]$items))
    )
  } else {
    history = data.frame(values = t(do.call(rbind,lapply(1:length(object$recorder$values),
                                                         function(x) object$recorder$values[[x]]$items)))
    )
  }

  nm = object$recorder$metric_names$items
  colnames(history) = nm[!nm %in% c('epoch','time')]

  if(nrow(history)==1) {
    history['epoch'] = 0
  } else {
    history['epoch'] = 0:(nrow(history)-1)
  }

  history = history[,c(which(colnames(history)=="epoch"),which(colnames(history)!="epoch"))]
  invisible(history)
}

#' @title Summary
#'
#' @description Print a summary of `m` using a output text width of `n` chars
#'
#'
#' @param m m
#'
#' @export
summary.fastai.tabular.learner.TabularLearner <- function(object) {

  object$summary()

}


#' @title Split by idx
#'
#' @description Split the data according to the indexes in `valid_idx`.
#'
#'
#' @param valid_idx valid_idx
#'
#' @export
split_by_idx <- function(object, valid_idx) {

  if(is.list(valid_idx)) {
    stop("Only vectors can be passed", call. = FALSE)
  } else if(is.vector(valid_idx)) {
    object$split_by_idx(as.integer(valid_idx))
  } else {
    stop("Pass the sequence of indices")
  }

}




#' @title Add test
#'
#' @description Add test set containing `items` with an arbitrary `label`.
#'
#'
#' @param items items
#' @param label label
#' @param tfms tfms
#' @param tfm_y tfm_y
#'
#' @export
add_test <- function(object, items, label = NULL, tfms = NULL, tfm_y = NULL) {

  args <- list(
    items = items,
    label = label,
    tfms = tfms,
    tfm_y = tfm_y
  )

  do.call(object$add_test, args)

}

#' @title Data collate
#'
#' @description Convert `batch` items to tensor data.
#'
#'
#' @param batch batch
#'
#' @export
data_collate <- function(object, batch) {

  if(missing(object) & missing(batch)) {
    vision$data_collate
  } else {
    object$data_collate(
      batch = as.integer(batch)
    )
  }

}



#' @title get_emb_sz
#'
#' @description Get default embedding size from `TabularPreprocessor` `proc` or the ones in `sz_dict`
#'
#' @details
#'
#' @param to to
#' @param sz_dict sz_dict
#'
#' @export
get_emb_sz <- function(to, sz_dict = NULL) {

  tabular$get_emb_sz(
    to = to,
    sz_dict = sz_dict
  )

}




