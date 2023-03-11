


#' @title RandomSplitter
#'
#' @description Randomly splits items.
#'
#' @param probs `Sequence` of probabilities that must sum to one. The length of
#' the `Sequence` is the number of groups to to split the items into.
#' @param seed Internal seed used for shuffling the items. Define this if you need reproducible results.
#' @return None
#' @export
icevision_RandomSplitter <- function(probs, seed = NULL) {

  args <- list(
    probs = probs,
    seed = seed
  )

  if(is.null(args$seed))
    args$seed <- NULL
  else
    args$seed <- as.integer(args$seed)

  do.call(icevision()$data$RandomSplitter, args)

}


#' @title FixedSplitter
#'
#' @description Split `ids` based on predefined splits.
#'
#' @param splits The predefined splits.
#' @return None
#' @export
icevision_FixedSplitter <- function(splits) {

  icevision()$data$FixedSplitter(
    splits = splits
  )

}



#' @title SingleSplitSplitter
#' @param ... arguments to pass
#'
#'
#' @return all items in a single group, without shuffling.
#' @export
icevision_SingleSplitSplitter = function(...) {
  args = list(...)

  if(length(args)>0) {
    do.call(icevision()$data$SingleSplitSplitter, args)
  } else {
    icevision()$data$SingleSplitSplitter
  }
}



#' @title IDMap
#'
#' @description Works like a dictionary that automatically assign values for new keys.
#
#'
#' @param initial_names initial_names
#' @return None
#' @export
IDMap <- function(initial_names = NULL) {

  icevision()$all$IDMap(
    initial_names = initial_names
  )

}





