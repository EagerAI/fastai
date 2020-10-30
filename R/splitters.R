

#' @title GrandparentSplitter
#'
#' @description Split `items` from the grand parent folder names (`train_name` and `valid_name`).
#'
#'
#' @param train_name train folder name
#' @param valid_name validation folder name
#' @return None
#' @export
GrandparentSplitter <- function(train_name = "train", valid_name = "valid") {

  fastai2$vision$all$GrandparentSplitter(
    train_name = train_name,
    valid_name = valid_name
  )

}


#' @title Parent_label
#'
#' @description Label `item` with the parent folder name.
#'
#'
#' @param o string, dir path
#' @return vector
#' @export
parent_label <- function(o) {

  fastai2$vision$all$parent_label(
    o = o
  )

}
