


#' @title COCOMetric
#'
#' @description Wrapper around [cocoapi evaluator](https://github.com/cocodataset/cocoapi)
#'
#' @details Calculates average precision. # Arguments metric_type: Dependent on the task you're solving. print_summary: If `TRUE`, prints a table with statistics. show_pbar: If `TRUE` shows pbar when preparing the data for evaluation.
#'
#' @param metric_type Dependent on the task you're solving.
#' @param print_summary If `TRUE`, prints a table with statistics.
#' @param show_pbar If `TRUE` shows pbar when preparing the data for evaluation.
#' @return None
#' @export
COCOMetric <- function(metric_type = 'bbox', print_summary = FALSE, show_pbar = FALSE) {

  args <- list(
    metric_type = metric_type,
    print_summary = print_summary,
    show_pbar = show_pbar
  )

  do.call(icevision()$all$COCOMetric, args)

}


