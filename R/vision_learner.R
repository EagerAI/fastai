#' @title has_pool_type
#'
#' @description Return `TRUE` if `m` is a pooling layer or has one in its children
#'
#'
#' @param m m
#'
#' @export
has_pool_type <- function(m) {

  vision$all$has_pool_type(
    m = m
  )

}

#' @title create_body
#'
#' @description Cut off the body of a typically pretrained `arch` as determined by `cut`
#'
#'
#' @param arch arch
#' @param n_in n_in
#' @param pretrained pretrained
#' @param cut cut
#'
#' @export
create_body <- function(arch, n_in = 3, pretrained = TRUE, cut = NULL) {

  vision$all$create_body(
    arch = arch,
    n_in = as.integer(n_in),
    pretrained = pretrained,
    cut = cut
  )

}

#' @title create_head
#'
#' @description Model head that takes `nf` features, runs through `lin_ftrs`, and out `n_out` classes.
#'
#'
#' @param nf nf
#' @param n_out n_out
#' @param lin_ftrs lin_ftrs
#' @param ps ps
#' @param concat_pool concat_pool
#' @param bn_final bn_final
#' @param lin_first lin_first
#' @param y_range y_range
#'
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


#' @title default_split
#'
#' @description Default split of a model between body and head
#'
#'
#' @param m m
#'
#' @export
default_split <- function(m) {

  vision$all$default_split(
    m = m
  )

}











