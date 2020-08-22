
#' @title Plot confusion matrix
#'
#' @importFrom highcharter hchart hc_yAxis hc_xAxis
#' @param object model
#' @param dataloader dataloaders object
#' @export
plot_confusion_matrix <- function(object, dataloader) {
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
}


#' @title Extract confusion matrix
#'
#' @importFrom highcharter hchart hc_yAxis hc_xAxis
#' @param object model
#' @param dataloader dataloaders object
#' @export
get_confusion_matrix <- function(object, dataloader) {
  interp = vision$all$ClassificationInterpretation$from_learner(object)

  conf=interp$confusion_matrix()
  conf=apply(conf, 2, as.integer)
  itms = dls$vocab$items$items
  colnames(conf)=itms
  rownames(conf)=itms
  conf
}




