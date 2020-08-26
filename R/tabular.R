
#' @title Accuracy
#'
#' @description Compute accuracy with `targ` when `pred` is bs * n_classes
#'
#'
#' @param inp inp
#' @param targ targ
#' @param axis axis
#'
#' @export
accuracy <- function(inp, targ, axis = -1) {

  if(missing(inp) && missing(targ)){
    tabular$accuracy
  } else {
    args <- list(inp = inp,
                 targ = targ,
                 axis = as.integer(axis)
    )
    do.call(tabular$accuracy,args)
  }

}

attr(accuracy,"py_function_name") <- "accuracy"



#' @title TabularDataTable
#'
#' @description A `Tabular` object with transforms
#'
#'
#' @param df df
#' @param procs procs
#' @param cat_names cat_names
#' @param cont_names cont_names
#' @param y_names y_names
#' @param y_block y_block
#' @param splits splits
#' @param do_setup do_setup
#' @param device device
#' @param inplace inplace
#' @param reduce_memory reduce_memory
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

#' @title Dataloaders
#'
#' @description Get a `DataLoaders`
#'
#'
#' @param object object
#' @param bs bs
#' @param val_bs val_bs
#' @param shuffle_train shuffle_train
#' @param n n
#' @param ... parameters to pass
#'
#' @export
dataloaders <- function(object, bs = 64, val_bs = NULL, shuffle_train = TRUE, n = NULL, ...) {

  args <- list(
    bs = as.integer(bs),
    val_bs = val_bs,
    shuffle_train = shuffle_train,
    n = n,
    ...
  )

  do.call(object$dataloaders, args)

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
#' @param dls dls
#' @param layers layers
#' @param emb_szs emb_szs
#' @param config config
#' @param n_out n_out
#' @param y_range y_range
#' @param loss_func loss_func
#' @param opt_func opt_func
#' @param lr lr
#' @param splitter splitter
#' @param cbs cbs
#' @param metrics metrics
#' @param path path
#' @param model_dir model_dir
#' @param wd wd
#' @param wd_bn_bias wd_bn_bias
#' @param train_bn train_bn
#' @param moms moms
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
fit.fastai.tabular.learner.TabularLearner <- function(object, n_epoch, lr = 1e-2, wd = NULL, callbacks = NULL) {

  args <- list(
    n_epoch = as.integer(n_epoch),
    lr = lr,
    wd = wd,
    callbacks = callbacks
  )

  # fit the model
  do.call(object$fit, args)

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




