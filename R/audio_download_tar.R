



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

  download.file(paste(fastaudio()$core$all$URLs$SPEAKERS10,sep = ''),
                destfile = paste(filename,'.tgz',sep = ''))

  if(untar) {
    if(!dir.exists('SPEAKERS10')) {
      dir.create('SPEAKERS10')
    }
    untar(paste(filename,'.tgz',sep = ''),exdir = 'SPEAKERS10')
  }

}


#' @title SPEECHCOMMANDS dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download SPEECHCOMMANDS dataset
#' @return None
#' @examples
#' \dontrun{
#'
#' URLs_SPEECHCOMMANDS()
#'
#' }
#'
#' @export
URLs_SPEECHCOMMANDS <- function(filename = 'SPEECHCOMMANDS', untar = TRUE) {

  download.file(paste("https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02",
                      ".tar.gz",
                      sep = ''),
                destfile = paste(filename,'.tar.gz',sep = ''))

  if(untar) {
    if(!dir.exists('SPEECHCOMMANDS')) {
      dir.create('SPEECHCOMMANDS')
    }
    untar(paste(filename,'.tar.gz',sep = ''),exdir = 'SPEECHCOMMANDS')
  }

}
