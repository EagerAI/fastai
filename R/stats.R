
plot_confusion_matrix <- function(object, dataloader) {
  if(inherits(object,"fastai.learner.Learner")) {
    interp = vision$all$ClassificationInterpretation$from_learner(object)

    conf=interp$confusion_matrix()
    conf=apply(conf, 2, as.integer)
    itms = dls$vocab$items$items
    colnames(conf)=itms
    rownames(conf)=itms

    hchart(conf,label=TRUE) %>%
      hc_yAxis(title = list(text='Actual')) %>%
      hc_xAxis(title = list(text='Predicted'),
               labels = list(rotation=-90))
  } else if (inherits(object,"fastai.tabular.learner.TabularLearner")) {
    conf = tabular$ClassificationInterpretation$from_learner(model)$confusion_matrix()
    colnames(conf)=dls$vocab$items$items
    rownames(conf)=dls$vocab$items$items
    hchart(conf,label=TRUE) %>%
      hc_yAxis(title = list(text='Actual')) %>%
      hc_xAxis(title = list(text='Predicted'),
               labels = list(rotation=-90))
  }

}


#' @title Extract confusion matrix
#'
#' @param object model
#' @param dataloader dataloaders object
#' @export
get_confusion_matrix <- function(object, dataloader) {
  if(inherits(object,"fastai.learner.Learner")) {
    interp = vision$all$ClassificationInterpretation$from_learner(object)

    conf=interp$confusion_matrix()
    conf=apply(conf, 2, as.integer)
    itms = dls$vocab$items$items
    colnames(conf)=itms
    rownames(conf)=itms
    conf
  } else if (inherits(object,"fastai.tabular.learner.TabularLearner")) {
    tabular$ClassificationInterpretation$from_learner(model)$confusion_matrix()
  }
}

#' @title Most_confused
#' @importFrom data.table rbindlist
#' @description Sorted descending list of largest non-diagonal entries of confusion matrix, presented as actual, predicted, number of occurrences.
#'
#' @param object learner
#'
#' @param min_val min_val
#'
#' @export
most_confused <- function(object, min_val = 1) {

  res = rbindlist(
    object$most_confused(
    min_val = as.integer(min_val)
  )
  )
  colnames(res) = c('Actual','Predicted','Confused_n')
}


#' @title lr_find
#'
#' @description Launch a mock training to find a good learning rate, return lr_min, lr_steep if `suggestions` is TRUE
#'
#' @param objetc learner
#' @param start_lr start_lr
#' @param end_lr end_lr
#' @param num_it num_it
#' @param stop_div stop_div
#' @param show_plot show_plot
#' @param suggestions suggestions
#'
#' @export
lr_find <- function(object, start_lr = 1e-07, end_lr = 10, num_it = 100,
                    stop_div = TRUE, suggestions = TRUE) {

  args <- list(
    start_lr = start_lr,
    end_lr = as.integer(end_lr),
    num_it = as.integer(num_it),
    stop_div = stop_div,
    show_plot = FALSE,
    suggestions = suggestions
  )

  do.call(object$lr_find, args)

}

#' @title lr_find
#'
#' @description Launch a mock training to find a good learning rate, return lr_min, lr_steep if `suggestions` is TRUE
#'
#' @param object learner
#'
#' @export
lr_find_ <- function(object) {

  losses = object$recorder$losses
  losses = unlist(lapply(1:length(losses),function(x) losses[[x]]$numpy()))
  lrs = object$recorder$lrs

  data.frame(lr_rates = lrs,
             losses = losses)

}


#' @title Accuracy
#'
#' @description Compute accuracy with `targ` when `pred` is bs * n_classes
#'
#'
#' @param inp inp
#' @param targ targ
#' @param axis axis
#'
#' @export
accuracy <- function(inp, targ, axis = -1) {

  if(missing(inp) && missing(targ)){
    tabular$accuracy
  } else {
    args <- list(inp = inp,
                 targ = targ,
                 axis = as.integer(axis)
    )
    do.call(tabular$accuracy,args)
  }

}

attr(accuracy,"py_function_name") <- "accuracy"

#' @title Perplexity
#'
#'
#' @param ... parameters to pass
#'
#' @export
Perplexity <- function(...) {
  invisible(text$Perplexity(...))
}


#' @title One batch
#'
#' @param convert to R matrix
#' @param object dataloader
#'
#' @export
one_batch <- function(object, convert = TRUE) {
  obj = object$one_batch()

  if(inherits(dls,'fastai.tabular.data.TabularDataLoaders')) {
    obj
  } else {
    if(convert) {
      bs = object$bs - 1
      obj[[1]] = lapply(0:bs, function(x) aperm(obj[[1]][[x]]$cpu()$numpy(), c(2,3,1)))
      indices = obj[[2]]$cpu()$numpy()
      list(obj[[1]],indices)
    } else {
      obj
    }
  }

}

#' @title Summary
#' @param object model
#' @export
summary.fastai.learner.Learner <- function(object) {
  object$summary()
}



