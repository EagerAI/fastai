
#' @title Parse
#'
#' @description Loops through all data points parsing the required fields.
#'
#'
#' @param data_splitter How to split the parsed data, defaults to a [0.8, 0.2] random split.
#' @param idmap Maps from filenames to unique ids, pass an `IDMap()` if you need this information.
#' @param autofix autofix
#' @param show_pbar Whether or not to show a progress bar while parsing the data.
#' @param cache_filepath Path to save records in pickle format. Defaults to NULL, e.g. if the user does not specify a path, no saving nor loading happens.
#' @return A list of records for each split defined by data_splitter.
#' @export
parse <- function(data_splitter = NULL, idmap = NULL, autofix = TRUE,
                  show_pbar = TRUE, cache_filepath = NULL) {

  args <- list(
    data_splitter = data_splitter,
    idmap = idmap,
    autofix = autofix,
    show_pbar = show_pbar,
    cache_filepath = cache_filepath
  )

  if(is.null(args$data_splitter))
    args$data_splitter <- NULL

  if(is.null(args$idmap))
    args$idmap <- NULL

  if(is.null(args$cache_filepath))
    args$cache_filepath <- NULL

  do.call(icevision()$parsers$Parser$parse,args)

}

#' @title Faster RCNN
#' @description Parser with required mixins for Faster RCNN.
#'
#' @param ... arguments to pass
#' @return None
#' @export
parsers_FasterRCNN = function(...) {
  icevision()$parsers$FasterRCNN(...)
}

#' @title Mask RCNN
#'
#' @description Parser with required mixins for Mask RCNN.
#' @param ... arguments to pass
#' @return None
#' @export
parsers_MaskRCNN = function(...) {
  icevision()$parsers$MaskRCNN(...)
}


#' @title Imageid Mixin
#'
#' @description Adds imageid method to parser
#' @param ... arguments to pass
#' @return None
#' @export
parsers_ImageidMixin = function(...) {
  icevision()$parsers$mixins$ImageidMixin(...)
}


#' @title FilepathMixin
#'
#' @description Adds filepath method to parser
#' @param ... arguments to pass
#' @return None
#' @export
parsers_FilepathMixin = function(...) {
  icevision()$parsers$mixins$FilepathMixin(...)
}


#' @title SizeMixin
#'
#' @description Adds image_width_height method to parser
#' @param ... arguments to pass
#' @return None
#' @export
parsers_SizeMixin = function(...) {
  icevision()$parsers$mixins$SizeMixin(...)
}


#' @title LabelsMixin
#'
#' @description Adds labels method to parser
#' @param ... arguments to pass
#' @return None
#' @export
parsers_LabelsMixin = function(...) {
  icevision()$parsers$mixins$LabelsMixin(...)
}

#' @title BBoxesMixin
#'
#' @description Adds bboxes method to parser
#' @param ... arguments to pass
#' @return None
#' @export
parsers_BBoxesMixin = function(...) {
  icevision()$parsers$mixins$BBoxesMixin(...)
}

#' @title MasksMixin
#'
#' @description Adds masks method to parser
#' @param ... arguments to pass
#' @return None
#' @export
parsers_MasksMixin = function(...) {
  icevision()$parsers$mixins$MasksMixin(...)
}

#' @title AreasMixin
#'
#' @description Adds areas method to parser
#' @param ... arguments to pass
#' @return None
#' @export
parsers_AreasMixin = function(...) {
  icevision()$parsers$mixins$AreasMixin(...)
}

#' @title IsCrowdsMixin
#'
#' @description Adds iscrowds method to parser
#' @param ... arguments to pass
#' @return None
#' @export
parsers_IsCrowdsMixin = function(...) {
  icevision()$parsers$mixins$IsCrowdsMixin(...)
}










