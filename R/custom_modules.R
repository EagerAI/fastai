

#' @title Net
#'
#'
#' @description Net model from Migrating_Pytorch
#'
#' @export
Net = function() {
  migrating_pytorch$Net()
}



#' @title train_loader
#'
#' @description Data loader. Combines a dataset and a sampler, and provides an iterable over
#'
#' @details the given dataset. The :class:`~torch.utils.data.DataLoader` supports both map-style and
#' iterable-style datasets with single- or multi-process loading, customizing
#' loading order and optional automatic batching (collation) and memory pinning. See :py:mod:`torch.utils.data` documentation page for more details.
#'
#'
#' @section The :class:`~torch.utils.data.DataLoader` supports both map-style and:
#' iterable-style datasets with single- or multi-process loading, customizing loading order and optional automatic batching (collation) and memory pinning.
#'
#' @export
train_loader = function() {
  invisible(migrating_pytorch$train_loader)
}


#' @title test_loader
#'
#' @description Data loader. Combines a dataset and a sampler, and provides an iterable over
#'
#' @details the given dataset. The :class:`~torch.utils.data.DataLoader` supports both map-style and
#' iterable-style datasets with single- or multi-process loading, customizing
#' loading order and optional automatic batching (collation) and memory pinning. See :py:mod:`torch.utils.data` documentation page for more details.
#'
#'
#' @section The :class:`~torch.utils.data.DataLoader` supports both map-style and:
#' iterable-style datasets with single- or multi-process loading, customizing loading order and optional automatic batching (collation) and memory pinning.
#'
#' @export
test_loader = function() {
  migrating_pytorch$test_loader
}


#' @title get_data_loaders
#'
#'
#' @param train_batch_size train_batch_size
#' @param val_batch_size val_batch_size
#'
#' @export
get_data_loaders <- function(train_batch_size, val_batch_size) {

  migrating_ignite$get_data_loaders(
    train_batch_size = as.integer(train_batch_size),
    val_batch_size = as.integer(val_batch_size)
  )


}


#' @title Lit Model
#'
#'
#' @export
LitModel = function() {
  migrating_lightning$LitModel()
}








