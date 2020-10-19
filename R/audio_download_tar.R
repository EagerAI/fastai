



#' @title SPEAKERS10 dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download SPEAKERS10 dataset
#' @return None
#' @examples
#' \dontrun{
#'
#' URLs_SPEAKERS10()
#'
#' }
#'
#' @export
URLs_SPEAKERS10 <- function(filename = 'SPEAKERS10', untar = TRUE) {

  download.file(paste(fastaudio$core$all$URLs$SPEAKERS10,sep = ''),
                destfile = paste(filename,'.tgz',sep = ''))

  if(untar)
    untar(paste(filename,'.tgz',sep = ''))

}




