#' @title CollabDataLoaders_from_df
#'
#' @description Create a `DataLoaders` suitable for collaborative filtering from `ratings`.
#'
#' @param ratings ratings
#' @param valid_pct valid_pct
#' @param user_name user_name
#' @param item_name item_name
#' @param rating_name rating_name
#' @param seed seed
#' @param path path
#' @param bs bs
#' @param val_bs val_bs
#' @param shuffle_train shuffle_train
#' @param device device
#'
#' @export
CollabDataLoaders_from_df <- function(ratings, valid_pct = 0.2, user_name = NULL,
                                      item_name = NULL, rating_name = NULL, seed = NULL,
                                      path = ".", bs = 64, val_bs = NULL, shuffle_train = TRUE,
                                      device = NULL) {

  args <- list(
    ratings = ratings,
    valid_pct = valid_pct,
    user_name = user_name,
    item_name = item_name,
    rating_name = rating_name,
    seed = seed,
    path = path,
    bs = as.integer(bs),
    val_bs = val_bs,
    shuffle_train = shuffle_train,
    device = device
  )

  if(!is.null(seed)) {
    args$seed <- as.integer(args$seed)
  }

  if(!is.null(val_bs)) {
    args$val_bs <- as.integer(args$val_bs)
  }

  do.call(collab$CollabDataLoaders$from_df, args)

}


#' @title CollabDataLoaders_from_dblock
#'
#' @description Create a dataloaders from a given `dblock`
#'
#' @param dblock dblock
#' @param source source
#' @param path path
#' @param bs bs
#' @param val_bs val_bs
#' @param shuffle_train shuffle_train
#' @param device device
#'
#' @export
CollabDataLoaders_from_dblock <- function(dblock, source, path = ".", bs = 64,
                                          val_bs = NULL, shuffle_train = TRUE,
                                          device = NULL) {

  args <- list(
    dblock = dblock,
    source = source,
    path = path,
    bs = as.integer(bs),
    val_bs = val_bs,
    shuffle_train = shuffle_train,
    device = device
  )

  if(!is.null(val_bs)) {
    args$val_bs <- as.integer(args$val_bs)
  }

  do.call(collab$CollabDataLoaders$from_dblock, args)

}



#' @title Collab_learner
#'
#' @description Create a Learner for collaborative filtering on `dls`.
#'
#'
#' @param dls dls
#' @param n_factors n_factors
#' @param use_nn use_nn
#' @param emb_szs emb_szs
#' @param layers layers
#' @param config config
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
collab_learner <- function(dls, n_factors = 50, use_nn = FALSE,
                           emb_szs = NULL, layers = NULL, config = NULL,
                           y_range = NULL, loss_func = NULL, opt_func = Adam(),
                           lr = 0.001, splitter = trainable_params(), cbs = NULL,
                           metrics = NULL, path = NULL, model_dir = "models",
                           wd = NULL, wd_bn_bias = FALSE, train_bn = TRUE,
                           moms = list(0.95, 0.85, 0.95)) {

  python_function_result <- fastai2$collab$collab_learner(
    dls = dls,
    n_factors = as.integer(n_factors),
    use_nn = use_nn,
    emb_szs = emb_szs,
    layers = layers,
    config = config,
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

}


#' @title trainable_params
#'
#' @description Return all trainable parameters of `m`
#'
#'
#' @param m m
#'
#' @export
trainable_params <- function(m) {

 if(missing(m)) {
   collab$trainable_params
 } else {
   collab$trainable_params(
     m = m
   )
 }

}


#' @title Get weights
#'
#' @description Weight for item or user (based on `is_item`) for all in `arr`
#'
#' @param object object
#' @param arr arr
#' @param is_item is_item
#' @param convert to matrix
#' @export
get_weights <- function(object, arr, is_item = TRUE, convert = FALSE) {

  if(convert)
    learn$model$weight(arr = arr,is_item = is_item)$numpy()
  else
    learn$model$weight(arr = arr,is_item = is_item)

}



#' @title Get bias
#'
#' @description Bias for item or user (based on `is_item`) for all in `arr`
#'
#' @param object object
#' @param arr arr
#' @param is_item is_item
#' @param convert to matrix
#' @export
get_bias <- function(object, arr, is_item = TRUE, convert = TRUE) {

  if(convert)
    learn$model$bias(arr = arr,is_item = is_item)$numpy()
  else
    learn$model$bias(arr = arr,is_item = is_item)

}


#' @title Pca
#'
#' @description Compute PCA of `x` with `k` dimensions.
#'
#'
#' @param object object
#' @param k k
#' @param convert to matrix
#' @export
pca <- function(object, k = 3, convert = TRUE) {

  if(convert){
    result = object$pca(
      k = as.integer(k)
    )
    t(result$t()$numpy())
  } else {
    object$pca(
      k = as.integer(k)
    )
  }

}

