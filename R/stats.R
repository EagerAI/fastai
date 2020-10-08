
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
#' @return matrix
#' @export
get_confusion_matrix <- function(object) {
  interp = vision$all$ClassificationInterpretation$from_learner(object)

  conf = interp$confusion_matrix()
  conf = apply(conf, 2, as.integer)
  itms = interp$vocab$items$items
  colnames(conf)=itms
  rownames(conf)=itms
  conf
}

#' @title Most_confused
#' @description Sorted descending list of largest non-diagonal entries of confusion matrix,
#' presented as actual, predicted, number of occurrences.
#'
#' @param object interpret object
#'
#' @param min_val minimum value
#' @return data.frame
#' @export
most_confused <- function(object, min_val = 1) {

  res = rbind(
    object$most_confused(
    min_val = as.integer(min_val)
  )
  )

  res = as.data.frame(res)

  colnames(res) = c('Actual','Predicted','Confused_n')
}


#' @title Lr_find
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
#' @return data frame
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

  print(do.call(object$lr_find, args))

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
#' @param inp predictions
#' @param targ actuals
#' @param axis axis
#' @return tensor
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
#' @return None
#' @export
Perplexity <- function(...) {
  invisible(text$Perplexity(...))
}

attr(Perplexity,"py_function_name") <- "Perplexity"

#' @title One batch
#'
#' @param convert to R matrix
#' @param object data loader
#' @return tensor
#' @export
one_batch <- function(object, convert = FALSE) {
  obj = object$one_batch()

  if(inherits(obj,'fastai.tabular.data.TabularDataLoaders')) {
    obj
  } else {
    if(convert) {
      if(length(dim(obj[[1]]$cpu()$numpy()))>2) {
        bs = object$bs - 1
        obj[[1]] = lapply(0:bs, function(x) aperm(obj[[1]][[x]]$cpu()$numpy(), c(2,3,1)))
        indices = obj[[2]]$cpu()$numpy()
        list(obj[[1]],indices)
      } else {
        res = lapply(1:length(obj), function(x) obj[[x]]$cpu()$numpy())
        res
      }
    } else {
      obj
    }
  }

}

#' @title Summary
#' @param object model
#' @return None
#' @export
summary.fastai.learner.Learner <- function(object) {
  object$summary()
}



#' @title Get_files
#'
#' @description Get all the files in `path` with optional `extensions`, optionally with `recurse`, only in `folders`, if specified.
#'
#'
#' @param path path
#' @param extensions extensions
#' @param recurse recurse
#' @param folders folders
#' @param followlinks followlinks
#' @return list
#' @export
get_files <- function(path, extensions = NULL, recurse = TRUE, folders = NULL, followlinks = TRUE) {

  vision$all$get_files(
    path = path,
    extensions = extensions,
    recurse = recurse,
    folders = folders,
    followlinks = followlinks
  )

}



#' @title Parallel
#'
#' @description Applies `func` in parallel to `items`, using `n_workers`
#'
#'
#' @param f file names
#' @param items items
#' @param ... additional arguments
#' @return None
#' @export
parallel <- function(f, items, ...) {

  tabular$parallel(
    f = f,
    items = items,
    ...
  )

}

