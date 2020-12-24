

#' @title Dataset
#'
#' @description Container for a list of records and transforms.
#'
#' @details Steps each time an item is requested (normally via directly indexing the `Dataset`):
#' Grab a record from the internal list of records.
#' Prepare the record (open the image, open the mask, add metadata).
#' Apply transforms to the record.
#'
#' @param records A list of records.
#' @param tfm Transforms to be applied to each item.
#' @return None
#' @export
icevision_Dataset <- function(records, tfm = NULL) {

  args = list(
    records = records,
    tfm = tfm
  )

  if(is.null(args$tfm))
    args$tfm <- NULL

  do.call(icevision()$data$dataset$Dataset, args)

}


#' @title Icevision Dataset from images
#'
#' @description Creates a `Dataset` from a list of images.
#'
#' @param images `Sequence` of images in memory (numpy arrays).
#' @param tfm Transforms to be applied to each item.
#'
#' @param ... additional arguments
#' @return None
#'
#' @export
icevision_Dataset_from_images <- function(images, tfm = NULL, ...) {

  args <- list(
    images = images,
    tfm = tfm,
    ...
  )

  if(is.null(args$tfm))
    args$tfm <- NULL

  do.call(icevision()$data$dataset$Dataset$from_images, args)

}









