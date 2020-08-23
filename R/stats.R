
plot_confusion_matrix <- function(object, dataloader) {
  if(inherits(object,"fastai2.learner.Learner")) {
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
  } else if (inherits(object,"fastai2.tabular.learner.TabularLearner")) {
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
#' @importFrom highcharter hchart hc_yAxis hc_xAxis
#' @param object model
#' @param dataloader dataloaders object
#' @export
get_confusion_matrix <- function(object, dataloader) {
  if(inherits(object,"fastai2.learner.Learner")) {
    interp = vision$all$ClassificationInterpretation$from_learner(object)

    conf=interp$confusion_matrix()
    conf=apply(conf, 2, as.integer)
    itms = dls$vocab$items$items
    colnames(conf)=itms
    rownames(conf)=itms
    conf
  } else if (inherits(object,"fastai2.tabular.learner.TabularLearner")) {
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




