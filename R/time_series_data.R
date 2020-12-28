


#' @title TSBlock
#'
#' @description A TimeSeries Block to process one timeseries
#'
#'
#' @param ... parameters to pass
#' @return None
#' @export
TSBlock <- function(...) {

  args = list(...)

  if(length(args)>0) {
    do.call(tms()$data$TSBlock, args)
  } else {
    tms()$data$TSBlock
  }

}


#' @title TSeries
#'
#' @description Basic Time series wrapper
#'
#'
#' @param ... parameters to pass
#' @return None
#' @export
TSeries <- function(...) {

  args = list(...)

  if(length(args)>0) {
    do.call(tms()$data$TSeries, args)
  } else {
    tms()$data$TSeries
  }

}

#' @title Stack_train_valid
#'
#' @description Stack df_train and df_valid, adds `valid_col`=TRUE/FALSE for df_valid/df_train
#'
#'
#' @param df_train train data
#' @param df_valid validation data
#' @return data frame
#' @export
stack_train_valid <- function(df_train, df_valid) {

  tms()$data$stack_train_valid(
    df_train = df_train,
    df_valid = df_valid
  )

}



#' @title TSDataLoaders_from_dfs
#'
#' @description Create a DataLoader from a df_train and df_valid
#'
#' @param df_train train data
#' @param df_valid validation data
#' @param path path (optional)
#' @param x_cols predictors
#' @param label_col label/output column
#' @param y_block y_block
#' @param item_tfms item transformations
#' @param batch_tfms batch transformations
#' @param bs batch size
#' @param val_bs validation batch size
#' @param shuffle_train shuffle train data
#' @param device device name
#' @return None
#' @export
TSDataLoaders_from_dfs <- function(df_train, df_valid, path = ".", x_cols = NULL, label_col = NULL,
                     y_block = NULL, item_tfms = NULL, batch_tfms = NULL, bs = 64,
                     val_bs = NULL, shuffle_train = TRUE, device = NULL) {

  args <- list(
    df_train = df_train,
    df_valid = df_valid,
    path = path,
    x_cols = x_cols,
    label_col = label_col,
    y_block = y_block,
    item_tfms = item_tfms,
    batch_tfms = batch_tfms,
    bs = as.integer(bs),
    val_bs = val_bs,
    shuffle_train = shuffle_train,
    device = device
  )

  strings = c('x_cols', 'label_col',
              'y_block', 'item_tfms', 'batch_tfms', 'device')

  for(i in 1:length(strings)) {
    if(is.null(args[[strings[i]]]))
      args[[strings[i]]] <- NULL
  }

  if(!is.null(args$batch_tfms)) {
    args$batch_tfms <- unlist(args$batch_tfms)
  }

  if(is.null(args[['val_bs']])) {
    args[['val_bs']] <- NULL
  } else {
    args[['val_bs']] <- as.integer(args[['val_bs']])
  }


  do.call(tms()$data$TSDataLoaders$from_dfs, args)

}

