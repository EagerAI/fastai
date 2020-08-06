
#' @title Accuracy
#'
#' @description Computes accuracy with `targs` when `input` is bs * n_classes.
#'
#' @param input input
#' @param targs targs
#'
#' @export
accuracy <- function(input, targs) {


  if(missing(input) & missing(targs)) {
    tabular$accuracy
  } else {
    args <- list(
      input = input,
      targs = targs
    )

    do.call(tabular$accuracy, args)
  }

}

attr(accuracy,"py_function_name") <- "accuracy"





#' @title Tabular List from dataframe
#'
#' @description Get the list of inputs in the `col` of `path/csv_name`.
#'
#' @param cls cls
#' @param df df
#' @param cat_names cat_names
#' @param cont_names cont_names
#' @param procs procs
#'
#' @export
tabular_TabularList_from_df <- function(df, cat_names = NULL, cont_names = NULL, procs = NULL) {

  args <- list(
    df = df,
    cat_names = cat_names,
    cont_names = cont_names,
    procs = procs
  )
  do.call(tabular$TabularList$from_df, args)

}


#' @title Tabular learner
#'
#' @description Get a `Learner` using `data`, with `metrics`,
#' including a `TabularModel` created using the remaining params.
#'
#'
#' @param data data
#' @param layers layers
#' @param emb_szs emb_szs
#' @param metrics metrics
#' @param ps ps
#' @param emb_drop emb_drop
#' @param y_range y_range
#' @param use_bn use_bn
#'
#' @export
tabular_learner <- function(data, layers, emb_szs = NULL, metrics = NULL,
                            path = NULL, callback_fns = NULL,
                            ps = NULL, emb_drop = 0.0, y_range = NULL, use_bn = TRUE) {

  args <- list(
    data = data,
    layers = layers,
    emb_szs = emb_szs,
    metrics = metrics,
    path = path,
    ps = ps,
    emb_drop = emb_drop,
    y_range = y_range,
    use_bn = use_bn
  )

  if (is.null(args$callback_fns)) {
    args$callback_fns = list(CSVLogger())
  } else {
    args$callback_fns = ifelse(!is.list(callback_fns),list(callback_fns),callback_fns)
    args$callback_fns = args$callback_fns %>% append(CSVLogger())
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
fit.fastai.basic_train.Learner <- function(object, epochs, lr = 1e-2, wd = NULL, callbacks = NULL) {

  args <- list(
    epochs = as.integer(epochs),
    lr = lr,
    wd = wd,
    callbacks = callbacks
  )

  # fit the model
  history <- do.call(object$fit, args)

  # get csv file
  history <- object$csv_logger$read_logged_file()

  invisible(fastai_history(history))

}

#' @title Summary
#'
#' @description Print a summary of `m` using a output text width of `n` chars
#'
#'
#' @param m m
#'
#' @export
summary.fastai.basic_train.Learner <- function(m, n = 70) {

  m$summary(
    n = as.integer(n)
  )

}

#' @title Plot history
#' @export
fastai_history <- function(history) {

  return(
    structure(class = "fastai_training_history", list(
      train_loss = history$train_loss,
      valid_loss = history$valid_loss
    ))
  )

}

#' @export
plot.fastai_training_history <- function(history) {

  history = fastai_history(history)
  plot(history$train_loss, history$valid_loss)
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

#' @title Label from dataframe
#'
#' @description Pass label column name from dataframe
#'
#'
#' @param ... arguments to pass
#'
#' @export
label_from_df <- function(object,...) {
  args = list(...)
  do.call(object$label_from_df,args)
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

#' @title Databunch
#'
#' @description Create an `DataBunch` from self, `path` will override `self.path`,
#' `kwargs` are passed to `DataBunch.create`.
#'
#' @details
#'
#' @param path path
#' @param bs bs
#' @param val_bs val_bs
#' @param num_workers num_workers
#' @param dl_tfms dl_tfms
#' @param device device
#' @param collate_fn collate_fn
#' @param no_check no_check
#'
#' @export
databunch <- function(object, path = NULL, bs = 64, val_bs = NULL, num_workers = 6,
                      dl_tfms = NULL, device = NULL, collate_fn = data_collate(), no_check = FALSE) {

  args <- list(
    path = path,
    bs = as.integer(bs),
    val_bs = as.integer(val_bs),
    num_workers = as.integer(num_workers),
    dl_tfms = dl_tfms,
    device = device,
    collate_fn = collate_fn,
    no_check = no_check
  )

  do.call(object$databunch, args)

}

#' @title TabularDataBunch
#'
#' @description Create a `DataBunch` suitable for tabular data.
#'
#'
#' @param train_dl train_dl
#' @param valid_dl valid_dl
#' @param fix_dl fix_dl
#' @param test_dl test_dl
#' @param device device
#' @param dl_tfms dl_tfms
#' @param path path
#' @param collate_fn collate_fn
#' @param no_check no_check
#'
#' @export
TabularDataBunch <- function(train_dl, valid_dl, fix_dl = NULL,
                             test_dl = NULL, device = NULL, dl_tfms = NULL,
                             path = getwd(), collate_fn = data_collate(), no_check = FALSE) {

  args <- list(
    train_dl = train_dl,
    valid_dl = valid_dl,
    fix_dl = fix_dl,
    test_dl = test_dl,
    device = device,
    dl_tfms = dl_tfms,
    path = path,
    collate_fn = collate_fn,
    no_check = no_check
  )

  do.call(tabular$TabularDataBunch, args)
}


#' @title from_df
#'
#' @description Create a `DataBunch` from `df` and `valid_idx` with `dep_var`. `kwargs` are passed to `DataBunch.create`.
#'
#' @param path path
#' @param df df
#' @param dep_var dep_var
#' @param valid_idx valid_idx
#' @param procs procs
#' @param cat_names cat_names
#' @param cont_names cont_names
#' @param classes classes
#' @param test_df test_df
#' @param bs bs
#' @param val_bs val_bs
#' @param num_workers num_workers
#' @param dl_tfms dl_tfms
#' @param device device
#' @param collate_fn collate_fn
#' @param no_check no_check
#'
#' @export
tabular_TabularDataBunch_from_df <- function(path, df, dep_var, valid_idx, procs = NULL, cat_names = NULL,
                    cont_names = NULL, classes = NULL, test_df = NULL,
                    bs = 64, val_bs = NULL, num_workers = 6L, dl_tfms = NULL,
                    device = NULL, collate_fn = data_collate(), no_check = FALSE) {

  args <- list(
    path = path,
    df = df,
    dep_var = dep_var,
    valid_idx = as.integer(valid_idx),
    procs = procs,
    cat_names = cat_names,
    cont_names = cont_names,
    classes = classes,
    test_df = test_df,
    bs = as.integer(bs),
    val_bs = as.integer(val_bs),
    num_workers = as.integer(num_workers),
    dl_tfms = dl_tfms,
    device = device,
    collate_fn = collate_fn,
    no_check = no_check
  )

  do.call(tabular$TabularDataBunch$from_df, args)

}


#' @title Get_emb_szs
#'
#' @description Return the default embedding sizes suitable for this data or takes the ones in `sz_dict`.
#'
#' @details
#'
#' @param sz_dict sz_dict
#'
#' @export
tabular_TabularList_get_emb_szs <- function(sz_dict = NULL) {

 tabular$TabularList$get_emb_szs(
    sz_dict = sz_dict
  )

}





