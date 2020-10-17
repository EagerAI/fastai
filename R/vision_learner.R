#' @title Has_pool_type
#'
#' @description Return `TRUE` if `m` is a pooling layer or has one in its children
#'
#'
#' @param m parameters
#' @return None
#' @export
has_pool_type <- function(m) {

  vision$all$has_pool_type(
    m = m
  )

}

#' @title Create_body
#'
#' @description Cut off the body of a typically pretrained `arch` as determined by `cut`
#'
#'
#' @param ... parameters to pass
#' @return None
#'
#' @examples
#'
#' \dontrun{
#'
#' encoder = create_body(resnet34(), pretrained = TRUE)
#'
#' }
#'
#' @export
create_body <- function(...) {

  args = list(...)

  do.call(fastai2$vision$all$create_body, args)

}

#' @title Create_head
#'
#' @description Model head that takes `nf` features, runs through `lin_ftrs`, and out `n_out` classes.
#'
#'
#' @param nf number of features
#' @param n_out number of out features
#' @param lin_ftrs linear features
#' @param ps parameter server
#' @param concat_pool concatenate pooling
#' @param bn_final batch normalization final
#' @param lin_first linear first
#' @param y_range y_range
#' @return None
#' @export
create_head <- function(nf, n_out, lin_ftrs = NULL, ps = 0.5, concat_pool = TRUE,
                        bn_final = FALSE, lin_first = FALSE, y_range = NULL) {

  vision$all$create_head(
    nf = as.integer(nf),
    n_out = as.integer(n_out),
    lin_ftrs = lin_ftrs,
    ps = ps,
    concat_pool = concat_pool,
    bn_final = bn_final,
    lin_first = lin_first,
    y_range = y_range
  )

}


#' @title Default_split
#'
#' @description Default split of a model between body and head
#'
#'
#' @param m parameters
#' @return None
#' @export
default_split <- function(m) {

  vision$all$default_split(
    m = m
  )

}











