
##################################################  Faster RSNN

#' @title Faster RSNN learner
#'
#' @description Fastai `Learner` adapted for Faster RCNN.
#'
#' @param dls `Sequence` of `DataLoaders` passed to the `Learner`.
#' The first one will be used for training and the second for validation.
#' @param model The model to train.
#' @param cbs Optional `Sequence` of callbacks.
#' @return model
#' @param ... learner_kwargs: Keyword arguments that will be internally passed to `Learner`.
#' @export
faster_rcnn_learner <- function(dls, model, cbs = NULL, ...) {

  args <- list(
    dls = dls,
    model = model,
    cbs = cbs,
    ...
  )


  do.call(icevision()$models$torchvision_models$faster_rcnn$fastai$learner, args)

}

#' @title Faster RSNN model
#'
#' @description FasterRCNN model implemented by torchvision.
#'
#' @param num_classes Number of classes.
#' @param backbone Backbone model to use. Defaults to a resnet50_fpn model.
#' @param remove_internal_transforms The torchvision model internally applies transforms like resizing and normalization, but we already do this at the `Dataset` level, so it's safe to remove those internal transforms.
#' @param pretrained Argument passed to `fastercnn_resnet50_fpn` if `backbone is NULL`. By default it is set to TRUE: this is generally used when training a new model (transfer learning). `pretrained = FALSE` is used during inference (prediction) for cases where the users have their own pretrained weights. **faster_rcnn_kwargs: Keyword arguments that internally are going to be passed to `torchvision.models.detection.faster_rcnn.FastRCNN`.
#' @return model
#' @export
faster_rcnn_model <- function(num_classes, backbone = NULL, remove_internal_transforms = TRUE, pretrained = TRUE) {

  args <- list(
    num_classes = as.integer(num_classes),
    backbone = backbone,
    remove_internal_transforms = remove_internal_transforms,
    pretrained = pretrained
  )

  if(is.null(args$backbone))
    args$backbone <- NULL


  do.call(icevision()$all$faster_rcnn$model, args)

}


##################################################  MaskRCNN learner


#' @title MaskRCNN learner
#'
#' @description Fastai `Learner` adapted for MaskRCNN.
#'
#' @param dls `Sequence` of `DataLoaders` passed to the `Learner`.
#' The first one will be used for training and the second for validation.
#' @param model The model to train.
#' @param cbs Optional `Sequence` of callbacks.
#' @return model
#' @param ... learner_kwargs: Keyword arguments that will be internally passed to `Learner`.
#' @export
mask_rcnn_learner <- function(dls, model, cbs = NULL, ...) {

  args <- list(
    dls = dls,
    model = model,
    cbs = cbs,
    ...
  )


  do.call(icevision()$models$torchvision_models$mask_rcnn$fastai$learner, args)

}


#' @title MaskRCNN model
#'
#' @description MaskRCNN model implemented by torchvision.
#'
#' @param num_classes Number of classes.
#' @param backbone Backbone model to use. Defaults to a resnet50_fpn model.
#' @param remove_internal_transforms The torchvision model internally applies transforms like resizing and normalization, but we already do this at the `Dataset` level, so it's safe to remove those internal transforms.
#' @param pretrained Argument passed to `maskrcnn_resnet50_fpn` if `backbone is NULL`. By default it is set to TRUE: this is generally used when training a new model (transfer learning). `pretrained = FALSE` is used during inference (prediction) for cases where the users have their own pretrained weights. **mask_rcnn_kwargs: Keyword arguments that internally are going to be passed to `torchvision.models.detection.mask_rcnn.MaskRCNN`.
#' @return model
#' @export
mask_rcnn_model <- function(num_classes, backbone = NULL, remove_internal_transforms = TRUE, pretrained = TRUE) {

  args <- list(
    num_classes = as.integer(num_classes),
    backbone = backbone,
    remove_internal_transforms = remove_internal_transforms,
    pretrained = pretrained
  )

  if(is.null(args$backbone))
    args$backbone <- NULL

  do.call(icevision()$all$mask_rcnn$model, args)

}


##################################################  EfficientDet learner


#' @title MaskRCNN learner
#'
#' @description Fastai `Learner` adapted for MaskRCNN.
#'
#' @param dls `Sequence` of `DataLoaders` passed to the `Learner`.
#' The first one will be used for training and the second for validation.
#' @param model The model to train.
#' @param cbs Optional `Sequence` of callbacks.
#' @return model
#' @param ... learner_kwargs: Keyword arguments that will be internally passed to `Learner`.
#' @export
efficientdet_learner <- function(dls, model, cbs = NULL, ...) {

  args <- list(
    dls = dls,
    model = model,
    cbs = cbs,
    ...
  )


  do.call(icevision()$models$efficientdet$fastai$learner, args)

}


#' @title Eficientdet model
#'
#' @description Creates the efficientdet model specified by `model_name`.
#'
#' @param model_name Specifies the model to create. For pretrained models, check [this](https://github.com/rwightman/efficientdet-pytorch#models) table.
#' @param num_classes Number of classes of your dataset (including background).
#' @param img_size Image size that will be fed to the model. Must be squared and divisible by 128.
#' @param pretrained If TRUE, use a pretrained backbone (on COCO).
#' @return model
#' @export
efficientdet_model <- function(model_name, num_classes, img_size, pretrained = TRUE) {

  args <- list(
    model_name = model_name,
    num_classes = as.integer(num_classes),
    img_size = as.integer(unlist(img_size)),
    pretrained = pretrained
  )

  do.call(icevision()$all$efficientdet$model, args)

}






