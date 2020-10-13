#' @title CollabDataLoaders_from_df
#'
#' @description Create a `DataLoaders` suitable for collaborative filtering from `ratings`.
#'
#' @param ratings ratings
#' @param valid_pct The random percentage of the dataset to set aside for validation (with an optional seed)
#' @param user_name The name of the column containing the user (defaults to the first column)
#' @param item_name The name of the column containing the item (defaults to the second column)
#' @param rating_name The name of the column containing the rating (defaults to the third column)
#' @param seed random seed
#' @param path The folder where to work
#' @param bs The batch size
#' @param val_bs The batch size for the validation DataLoader (defaults to bs)
#' @param shuffle_train If we shuffle the training DataLoader or not
#' @param device the device, e.g. cpu, cuda, and etc.
#' @return None
#'
#' @examples
#'
#' \dontrun{
#'
#' URLs_MOVIE_LENS_ML_100k()
#' c(user,item,title)  %<-% list('userId','movieId','title')
#' ratings = fread('ml-100k/u.data', col.names = c(user,item,'rating','timestamp'))
#' movies = fread('ml-100k/u.item', col.names = c(item, 'title', 'date', 'N', 'url',
#'                                                paste('g',1:19,sep = '')))
#' rating_movie = ratings[movies[, .SD, .SDcols=c(item,title)], on = item]
#' dls = CollabDataLoaders_from_df(rating_movie, seed = 42, valid_pct = 0.1, bs = 64,
#' item_name=title, path='ml-100k')
#'
#' }
#'
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
#' @param path The folder where to work
#' @param bs The batch size
#' @param val_bs The batch size for the validation DataLoader (defaults to bs)
#' @param shuffle_train If we shuffle the training DataLoader or not
#' @param device device
#' @return None
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
#' @param dls a data loader object
#' @param n_factors The number of factors
#' @param use_nn use_nn
#' @param emb_szs embedding size
#' @param layers list of layers
#' @param config configuration
#' @param y_range y_range
#' @param loss_func It can be any loss function you like. It needs to be one of fastai's if you want to use Learn.predict or Learn.get_preds, or you will have to implement special methods (see more details after the BaseLoss documentation).
#' @param opt_func The function used to create the optimizer
#' @param lr learning rate
#' @param splitter It is a function that takes self.model and returns a list of parameter groups (or just one parameter group if there are no different parameter groups).
#' @param cbs Cbs is one or a list of Callbacks to pass to the Learner.
#' @param metrics It is an optional list of metrics, that can be either functions or Metrics.
#' @param path The folder where to work
#' @param model_dir Path and model_dir are used to save and/or load models.
#' @param wd It is the default weight decay used when training the model.
#' @param wd_bn_bias It controls if weight decay is applied to BatchNorm layers and bias.
#' @param train_bn It controls if BatchNorm layers are trained even when they are supposed to be frozen according to the splitter.
#' @param moms The default momentums used in Learner.fit_one_cycle.
#' @return learner object
#' @examples
#'
#' \dontrun{
#'
#' URLs_MOVIE_LENS_ML_100k()
#' c(user,item,title)  %<-% list('userId','movieId','title')
#' ratings = fread('ml-100k/u.data', col.names = c(user,item,'rating','timestamp'))
#' movies = fread('ml-100k/u.item', col.names = c(item, 'title', 'date', 'N', 'url',
#'                                                paste('g',1:19,sep = '')))
#' rating_movie = ratings[movies[, .SD, .SDcols=c(item,title)], on = item]
#' dls = CollabDataLoaders_from_df(rating_movie, seed = 42, valid_pct = 0.1, bs = 64,
#' item_name=title, path='ml-100k')
#'
#' learn = collab_learner(dls, n_factors = 40, y_range=c(0, 5.5))
#'
#' learn %>% fit_one_cycle(1, 5e-3,  wd = 1e-1)
#'
#' }
#'
#' @export
collab_learner <- function(dls, n_factors = 50, use_nn = FALSE,
                           emb_szs = NULL, layers = NULL, config = NULL,
                           y_range = NULL, loss_func = NULL, opt_func = Adam(),
                           lr = 0.001, splitter = trainable_params(), cbs = NULL,
                           metrics = NULL, path = NULL, model_dir = "models",
                           wd = NULL, wd_bn_bias = FALSE, train_bn = TRUE,
                           moms = list(0.95, 0.85, 0.95)) {

  args = list(
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

  do.call(fastai2$collab$collab_learner, args)

}


#' @title Trainable_params
#'
#' @description Return all trainable parameters of `m`
#'
#' @param m trainable parameters
#' @return None
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
#' @param object extract weights
#' @param arr R data frame
#' @param is_item logical, is item
#' @param convert to R matrix
#' @return tensor
#'
#' @examples
#'
#' \dontrun{
#'
#' movie_w = learn %>% get_weights(top_movies, is_item = TRUE, convert = TRUE)
#'
#' }
#'
#' @export
get_weights <- function(object, arr, is_item = TRUE, convert = FALSE) {

  if(convert)
    object$model$weight(arr = arr,is_item = is_item)$numpy()
  else
    object$model$weight(arr = arr,is_item = is_item)

}



#' @title Get bias
#'
#' @description Bias for item or user (based on `is_item`) for all in `arr`
#'
#' @param object extract bias
#' @param arr R data frame
#' @param is_item logical, is item
#' @param convert to R matrix
#' @return tensor
#'
#' @examples
#'
#' \dontrun{
#'
#' movie_bias = learn %>% get_bias(top_movies, is_item = TRUE)
#'
#' }
#'
#' @export
get_bias <- function(object, arr, is_item = TRUE, convert = TRUE) {

  if(convert)
    object$model$bias(arr = arr,is_item = is_item)$numpy()
  else
    object$model$bias(arr = arr,is_item = is_item)

}


#' @title PCA
#'
#' @description Compute PCA of `x` with `k` dimensions.
#'
#'
#' @param object an object to apply PCA
#' @param k number of dimensions
#' @param convert to R matrix
#' @return tensor
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

