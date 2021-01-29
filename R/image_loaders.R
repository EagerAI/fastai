#' @title ImageDataLoaders from name regex
#'
#' @description Create from the name attrs of `fnames` in `path`s with re expression `pat`
#'
#' @param path The folder where to work
#' @param fnames folder names
#' @param pat an argument that requires regex
#' @param bs The batch size
#' @param val_bs The batch size for the validation DataLoader (defaults to bs)
#' @param shuffle_train If we shuffle the training DataLoader or not
#' @param device device name
#' @param item_tfms One or several transforms applied to the items before batching them
#' @param batch_tfms One or several transforms applied to the batches once they are formed
#' @param ... additional parameters to pass
#' @return None
#'
#' @examples
#'
#' \dontrun{
#'
#' URLs_PETS()
#'
#' path = 'oxford-iiit-pet'
#'
#' dls = ImageDataLoaders_from_name_re(
#' path, fnames, pat='(.+)_\\d+.jpg$',
#' item_tfms = RandomResizedCrop(460, min_scale=0.75), bs = 10,
#' batch_tfms = list(aug_transforms(size = 299, max_warp = 0),
#'                   Normalize_from_stats( imagenet_stats() )
#' ),
#' device = 'cuda'
#' )
#'
#' }
#'
#' @export
ImageDataLoaders_from_name_re <- function(path, fnames, pat, bs = 64,
                                          val_bs = NULL, shuffle_train = TRUE,
                                          device = NULL, item_tfms = NULL,
                                          batch_tfms = NULL,
                                          ...) {

  args <- list(
    path = path,
    fnames = fnames,
    pat = pat,
    bs = as.integer(bs),
    val_bs = val_bs,
    shuffle_train = shuffle_train,
    device = device,
    item_tfms = item_tfms,
    batch_tfms = batch_tfms,
    ...
  )

  if(!is.null(args$batch_tfms)) {
    args$batch_tfms <- unlist(args$batch_tfms)
  }

  if(!is.null(args$val_bs)) {
    args$val_bs <- as.integer(args$val_bs)
  }

  strings = c('val_bs', 'item_tfms', 'device', 'batch_tfms')

  for(i in 1:length(strings)) {
    if(is.null(args[[strings[i]]])) {
      args[[strings[i]]] <- NULL
    }
  }

  do.call(vision()$all$ImageDataLoaders$from_name_re,args)

}

#' @title Get image files
#'
#' @description Get image files in `path` recursively, only in `folders`, if specified.
#'
#'
#' @param path The folder where to work
#' @param recurse recursive path
#' @param folders folder names
#' @return None
#'
#' @examples
#'
#' \dontrun{
#'
#' URLs_PETS()
#'
#' path = 'oxford-iiit-pet'
#'
#' path_img = 'oxford-iiit-pet/images'
#' fnames = get_image_files(path_img)
#'
#' }
#'
#' @export
get_image_files <- function(path, recurse = TRUE, folders = NULL) {



  if(missing(path)) {
    invisible(vision()$all$get_image_files)
  } else {
    args <- list(
      path = path,
      recurse = recurse,
      folders = folders
    )
    do.call(vision()$all$get_image_files, args)
  }


}

#' @title Fit one cycle
#'
#' @param object model
#' @param ... parameters to pass, e.g. lr, n_epoch, wd, and etc.
#' @return None
#' @export
fit_one_cycle <- function(object, ...) {

  args = list(...)

  if(!is.null(args[['n_epoch']])) {
    args[['n_epoch']] = as.integer(args[['n_epoch']])
  }

  if(!is.null(args[[1]]) & is.null(names(args[[1]]))) {
    args[[1]] = as.integer(args[[1]])
  }

  do.call(object$fit_one_cycle, args)
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


#' @title ImageDataLoaders from folder
#'
#' @description Create from imagenet style dataset in `path` with `train` and `valid` subfolders (or provide `valid_pct`)
#'
#' @param path The folder where to work
#' @param train train data
#' @param valid validation data
#' @param valid_pct validion percentage
#' @param seed random seed
#' @param vocab vocabulary
#' @param item_tfms One or several transforms applied to the items before batching them
#' @param batch_tfms One or several transforms applied to the batches once they are formed
#' @param bs batch size
#' @param val_bs The batch size for the validation DataLoader (defaults to bs)
#' @param shuffle_train If we shuffle the training DataLoader or not
#' @param device device name
#' @param size image size
#' @param ... additional parameters to pass
#' @export
ImageDataLoaders_from_folder <- function(path, train = "train", valid = "valid",
                        valid_pct = NULL, seed = NULL, vocab = NULL,
                        item_tfms = NULL, batch_tfms = NULL, bs = 64,
                        val_bs = NULL, shuffle_train = TRUE, device = NULL,
                        size = NULL,
                        ...) {

  args <- list(
    path = path,
    train = train,
    valid = valid,
    valid_pct = valid_pct,
    seed = seed,
    vocab = vocab,
    item_tfms = item_tfms,
    batch_tfms = batch_tfms,
    bs = as.integer(bs),
    val_bs = val_bs,
    shuffle_train = shuffle_train,
    device = device,
    size = size,
    ...
  )
  if(!is.null(args$batch_tfms)) {
    args$batch_tfms <- unlist(args$batch_tfms)
  }

  if(!is.null(args$size)) {
    args$size = as.integer(args$size)
  }

  if(!is.null(args$val_bs)) {
    args$val_bs <- as.integer(args$val_bs)
  }

  do.call(vision()$all$ImageDataLoaders$from_folder, args)

}


#' @title ImageDataLoaders from csv
#'
#' @description Create from `path/csv_fname` using `fn_col` and `label_col`
#'
#' @param path The folder where to work
#' @param csv_fname csv file name
#' @param header header
#' @param delimiter delimiter
#' @param valid_pct validation percentage
#' @param seed random seed
#' @param fn_col column name
#' @param folder folder name
#' @param suff suff
#' @param label_col label column
#' @param label_delim label delimiter
#' @param y_block y_block
#' @param valid_col validation column
#' @param item_tfms One or several transforms applied to the items before batching them
#' @param batch_tfms One or several transforms applied to the batches once they are formed
#' @param bs batch size
#' @param val_bs The batch size for the validation DataLoader (defaults to bs)
#' @param shuffle_train If we shuffle the training DataLoader or not
#' @param size image size
#' @param device device name
#' @param ... additional parameters to pass
#' @return None
#' @export
ImageDataLoaders_from_csv <- function(path, csv_fname = "labels.csv", header = "infer",
                     delimiter = NULL, valid_pct = 0.2, seed = NULL, fn_col = 0,
                     folder = NULL, suff = "", label_col = 1, label_delim = NULL,
                     y_block = NULL, valid_col = NULL, item_tfms = NULL,
                     batch_tfms = NULL, bs = 64, val_bs = NULL,
                     size = NULL,
                     shuffle_train = TRUE, device = NULL,
                     ...) {

  args <- list(
    path = path,
    csv_fname = csv_fname,
    header = header,
    delimiter = delimiter,
    valid_pct = valid_pct,
    seed = seed,
    fn_col = as.integer(fn_col),
    folder = folder,
    suff = suff,
    label_col = as.integer(label_col),
    label_delim = label_delim,
    y_block = y_block,
    valid_col = valid_col,
    item_tfms = item_tfms,
    batch_tfms = batch_tfms,
    bs = as.integer(bs),
    val_bs = val_bs,
    shuffle_train = shuffle_train,
    device = device,
    size = size,
    ...
  )

  if(!is.null(args$batch_tfms)) {
    args$batch_tfms <- unlist(args$batch_tfms)
  }

  if(!is.null(size)) {
    args$size = as.integer(args$size)
  }

  if(!is.null(args$val_bs)) {
    args$val_bs = as.integer(args$val_bs)
  }

  do.call(vision()$all$ImageDataLoaders$from_csv, args)
}


#' @title ImageDataLoaders from dblock
#'
#' @description Create a dataloaders from a given `dblock`
#'
#' @param dblock dblock
#' @param source source folder
#' @param path The folder where to work
#' @param bs batch size
#' @param val_bs The batch size for the validation DataLoader (defaults to bs)
#' @param shuffle_train If we shuffle the training DataLoader or not
#' @param device device name
#' @param ... additional parameters to pass
#' @return None
#'
#' @export
ImageDataLoaders_from_dblock <- function(dblock, source, path = ".",
                                         bs = 64, val_bs = NULL,
                                         shuffle_train = TRUE, device = NULL,
                                         ...) {

  args <- list(
    dblock = dblock,
    source = source,
    path = path,
    bs = as.integer(bs),
    val_bs = val_bs,
    shuffle_train = shuffle_train,
    device = device,
    ...
  )

  if(!is.null(args$batch_tfms)) {
    args$batch_tfms <- unlist(args$batch_tfms)
  }

  if(!is.null(args$size)) {
    args$size = as.integer(args$size)
  }

  if(!is.null(args$val_bs)) {
    args$val_bs = as.integer(args$val_bs)
  }

  do.call(vision()$all$ImageDataLoaders$from_dblock, args)

}


#' @title ImageDataLoaders from df
#'
#' @description Create from `df` using `fn_col` and `label_col`
#'
#' @param df data frame
#' @param path The folder where to work
#' @param valid_pct validation percentage
#' @param seed random seed
#' @param fn_col column name
#' @param folder folder name
#' @param suff suff
#' @param label_col label column
#' @param label_delim label separator
#' @param y_block y_block
#' @param valid_col validation column
#' @param item_tfms One or several transforms applied to the items before batching them
#' @param batch_tfms One or several transforms applied to the batches once they are formed
#' @param bs batch size
#' @param val_bs The batch size for the validation DataLoader (defaults to bs)
#' @param shuffle_train shuffle_train
#' @param device device
#' @param ... additional parameters to pass
#' @return None
#' @export
ImageDataLoaders_from_df <- function(df, path = ".", valid_pct = 0.2, seed = NULL,
                    fn_col = 0, folder = NULL, suff = "", label_col = 1,
                    label_delim = NULL, y_block = NULL, valid_col = NULL,
                    item_tfms = NULL, batch_tfms = NULL, bs = 64,
                    val_bs = NULL, shuffle_train = TRUE, device = NULL,
                    ...) {

  args <- list(
    df = df,
    path = path,
    valid_pct = valid_pct,
    seed = seed,
    fn_col = as.integer(fn_col),
    folder = folder,
    suff = suff,
    label_col = as.integer(label_col),
    label_delim = label_delim,
    y_block = y_block,
    valid_col = valid_col,
    item_tfms = item_tfms,
    batch_tfms = batch_tfms,
    bs = as.integer(bs),
    val_bs = val_bs,
    shuffle_train = shuffle_train,
    device = device,
    ...
  )

  if(!is.null(args$batch_tfms)) {
    args$batch_tfms <- unlist(args$batch_tfms)
  }

  if(!is.null(args$size)) {
    args$size = as.integer(args$size)
  }

  if(!is.null(args$val_bs)) {
    args$val_bs = as.integer(args$val_bs)
  }

  do.call(vision()$all$ImageDataLoaders$from_df, args)

}


#' @title ImageDataLoaders from lists
#'
#' @description Create from list of `fnames` and `labels` in `path`
#'
#' @param path The folder where to work
#' @param fnames file names
#' @param labels labels
#' @param valid_pct validation percentage
#' @param seed random seed
#' @param y_block y_block
#' @param item_tfms One or several transforms applied to the items before batching them
#' @param batch_tfms One or several transforms applied to the batches once they are formed
#' @param bs batch size
#' @param val_bs The batch size for the validation DataLoader (defaults to bs)
#' @param shuffle_train If we shuffle the training DataLoader or not
#' @param device device name
#' @param ... additional parameters to pass
#' @return None
#' @export
ImageDataLoaders_from_lists <- function(path, fnames, labels, valid_pct = 0.2,
                       seed = NULL, y_block = NULL, item_tfms = NULL,
                       batch_tfms = NULL, bs = 64, val_bs = NULL,
                       shuffle_train = TRUE, device = NULL,
                       ...) {

  args <- list(
    path = path,
    fnames = fnames,
    labels = labels,
    valid_pct = valid_pct,
    seed = seed,
    y_block = y_block,
    item_tfms = item_tfms,
    batch_tfms = batch_tfms,
    bs = as.integer(bs),
    val_bs = val_bs,
    shuffle_train = shuffle_train,
    device = device,
    ...
  )

  if(!is.null(args$batch_tfms)) {
    args$batch_tfms <- unlist(args$batch_tfms)
  }

  if(!is.null(args$size)) {
    args$size = as.integer(args$size)
  }

  if(!is.null(args$val_bs)) {
    args$val_bs = as.integer(args$val_bs)
  }

  do.call(vision()$all$ImageDataLoaders$from_lists, args)

}



#' @title ImageDataLoaders from path function
#'
#' @description Create from list of `fnames` in `path`s with `label_func`
#' @param path The folder where to work
#' @param fnames file names
#' @param label_func label function
#' @param valid_pct The random percentage of the dataset to set aside for validation (with an optional seed)
#' @param seed random seed
#' @param item_tfms One or several transforms applied to the items before batching them
#' @param batch_tfms One or several transforms applied to the batches once they are formed
#' @param bs batch size
#' @param val_bs The batch size for the validation DataLoader (defaults to bs)
#' @param shuffle_train If we shuffle the training DataLoader or not
#' @param device device name
#' @param ... additional parameters to pass
#' @return None
#' @export
ImageDataLoaders_from_path_func <- function(path, fnames, label_func,
                                            valid_pct = 0.2, seed = NULL,
                                            item_tfms = NULL, batch_tfms = NULL,
                                            bs = 64, val_bs = NULL, shuffle_train = TRUE,
                                            device = NULL,
                                            ...) {

  args <- list(
    path = path,
    fnames = fnames,
    label_func = label_func,
    valid_pct = valid_pct,
    seed = seed,
    item_tfms = item_tfms,
    batch_tfms = batch_tfms,
    bs = as.integer(bs),
    val_bs = val_bs,
    shuffle_train = shuffle_train,
    device = device,
    ...
  )

  if(!is.null(args$batch_tfms)) {
    args$batch_tfms <- unlist(args$batch_tfms)
  }

  if(!is.null(args$size)) {
    args$size = as.integer(args$size)
  }

  if(!is.null(args$val_bs)) {
    args$val_bs = as.integer(args$val_bs)
  }

  do.call(vision()$all$ImageDataLoaders$from_path_func, args)

}


#' @title ImageDataLoaders from path re
#'
#' @description Create from list of `fnames` in `path`s with re expression `pat`
#'
#' @param path The folder where to work
#' @param fnames file names
#' @param pat an argument that requires regex
#' @param valid_pct The random percentage of the dataset to set aside for validation (with an optional seed)
#' @param seed random seed
#' @param item_tfms One or several transforms applied to the items before batching them
#' @param batch_tfms One or several transforms applied to the batches once they are formed
#' @param bs batch size
#' @param val_bs The batch size for the validation DataLoader (defaults to bs)
#' @param shuffle_train If we shuffle the training DataLoader or not
#' @param device device name
#' @param ... additional parameters to pass
#' @return None
#' @export
ImageDataLoaders_from_path_re <- function(path, fnames, pat, valid_pct = 0.2,
                                          seed = NULL, item_tfms = NULL,
                                          batch_tfms = NULL, bs = 64,
                                          val_bs = NULL, shuffle_train = TRUE,
                                          device = NULL, ...) {

  args <- list(
    path = path,
    fnames = fnames,
    pat = pat,
    valid_pct = valid_pct,
    seed = seed,
    item_tfms = item_tfms,
    batch_tfms = batch_tfms,
    bs = as.integer(bs),
    val_bs = val_bs,
    shuffle_train = shuffle_train,
    device = device,
    ...
  )

  if(!is.null(args$batch_tfms)) {
    args$batch_tfms <- unlist(args$batch_tfms)
  }

  if(!is.null(args$size)) {
    args$size = as.integer(args$size)
  }

  if(!is.null(args$val_bs)) {
    args$val_bs = as.integer(args$val_bs)
  }

  do.call(vision()$all$ImageDataLoaders$from_path_re, args)
}
