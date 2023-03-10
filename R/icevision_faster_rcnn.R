

#' @title Faster RCNN infer dataloader
#'
#' @description A `DataLoader` with a custom `collate_fn` that batches items as required for inferring the model.
#'
#' @param dataset Possibly a `Dataset` object, but more generally, any `Sequence` that returns records.
#' @param batch_tfms Transforms to be applied at the batch level. **dataloader_kwargs: Keyword arguments that will be internally passed to a Pytorch `DataLoader`. The parameter `collate_fn` is already defined internally and cannot be passed here.
#' @return None
#' @param ... additional arguments
#' @export
faster_rcnn_infer_dl <- function(dataset, batch_tfms = NULL, ...) {

  args <- list(
    dataset = dataset,
    batch_tfms = batch_tfms,
    ...
  )

  if(is.null(args$batch_tfms))
    args$batch_tfms <- NULL

  if(!is.null(args$batch_size))
    args$batch_size <- as.integer(args$batch_size)

  do.call(icevision()$faster_rcnn$infer_dl, args)

}



#' @title Faster RCNN predict dataloader
#' @param model model
#' @param infer_dl infer_dl
#' @param show_pbar show_pbar
#' @return None
#' @export
faster_rcnn_predict_dl <- function(model, infer_dl, show_pbar = TRUE) {

  icevision()$faster_rcnn$predict_dl(
    model = model,
    infer_dl = infer_dl,
    show_pbar = show_pbar
  )

}




