
#crap <- reticulate::import_from_path('crappify', path = 'fastaibuilt')

#' @title crappifier
#'
#'
#' @param path_lr path_lr
#' @param path_hr path_hr
#'
#' @export
crappifier <- function(path_lr, path_hr) {

  crap$crappifier(
    path_lr = path_lr,
    path_hr = path_hr
  )

}





