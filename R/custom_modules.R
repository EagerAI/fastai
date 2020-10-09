

#' @title Net
#'
#'
#' @description Net model from Migrating_Pytorch
#' @return model
#' @export
Net = function() {
  migrating_pytorch$Net()
}



#' @title Train_loader
#'
#' @description Data loader. Combines a dataset and a sampler, and provides an iterable over
#'
#' @details the given dataset. The :class:`~torch.utils.data.DataLoader` supports both map-style and
#' iterable-style datasets with single- or multi-process loading, customizing
#' loading order and optional automatic batching (collation) and memory pinning.
#'
#' @return loader
#' @export
train_loader = function() {
  invisible(migrating_pytorch$train_loader)
}


#' @title Test_loader
#'
#' @description Data loader. Combines a dataset and a sampler, and provides an iterable over
#'
#' @details the given dataset. The :class:`~torch.utils.data.DataLoader` supports both map-style and
#' iterable-style datasets with single- or multi-process loading, customizing
#' loading order and optional automatic batching (collation) and memory pinning. See :py:mod:`torch.utils.data` documentation page for more details.
#'
#' @return loader
#' @export
test_loader = function() {
  migrating_pytorch$test_loader
}


#' @title Get data loaders
#'
#'
#' @param train_batch_size train dataset batch size
#' @param val_batch_size validation dataset batch size
#' @return None
#' @export
get_data_loaders <- function(train_batch_size, val_batch_size) {

  migrating_ignite$get_data_loaders(
    train_batch_size = as.integer(train_batch_size),
    val_batch_size = as.integer(val_batch_size)
  )


}


#' @title Lit Model
#'
#' @return model
#' @export
LitModel = function() {
  migrating_lightning$LitModel()
}


#' @title Loaders
#' @description a loader from Catalyst
#' @return None
#' @export
loaders = function() {
  catalyst$loaders()
}


#' @title Catalyst model
#'
#' @return model
#' @export
catalyst_model = function() {
  catalyst$model
}









