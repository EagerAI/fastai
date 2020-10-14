#' @title Competition list files
#'
#' @description list files for competition
#'
#' @examples
#'
#' \dontrun{
#'
#' com_nm = 'titanic'
#' titanic_files = competition_list_files(com_nm)
#'
#'
#' }
#' @param competition the name of the competition
#' @return list of files
#' @export
competition_list_files <- function(competition) {

  kg$api$competition_list_files(
    competition = competition
  )

}


#' @title Competition download file
#'
#' @description download a competition file to a designated location, or use
#'
#' @examples
#'
#' \dontrun{
#'
#' com_nm = 'titanic'
#'
#' titanic_files = competition_list_files(com_nm)
#' titanic_files = lapply(1:length(titanic_files),
#'                       function(x) as.character(titanic_files[[x]]))
#'
#' str(titanic_files)
#'
#' if(!dir.exists(com_nm)) {
#'  dir.create(com_nm)
#' }
#'
#' # download via api
#' competition_download_files(competition = com_nm, path = com_nm, unzip = TRUE)
#'
#' }
#' @param competition the name of the competition
#' @param file_name the configuration file name
#' @param path a path to download the file to
#' @param force force the download if the file already exists (default FALSE)
#' @param quiet suppress verbose output (default is FALSE)
#' @return None
#' @export
competition_download_file <- function(competition, file_name, path = NULL, force = FALSE, quiet = FALSE) {

  kg$api$competition_download_file(
    competition = competition,
    file_name = file_name,
    path = path,
    force = force,
    quiet = quiet
  )

}

#' @title Competition download files
#'
#' @param competition the name of the competition
#' @param path a path to download the file to
#' @param force force the download if the file already exists (default FALSE)
#' @param quiet suppress verbose output (default is TRUE)
#' @param unzip unzip downloaded files
#' @return None
#' @export
competition_download_files <- function(competition, path = NULL, force = FALSE, quiet = FALSE,
                                       unzip = FALSE) {

  kg$api$competition_download_files(
    competition = competition,
    path = path,
    force = force,
    quiet = quiet
  )

  if(unzip) {
    fls = paste(competition, path, sep = '/')
    suppressWarnings(unzip(paste(fls,'.zip',sep = ''), exdir = competition))
  }

}




#' @title Competition leaderboard download
#'
#' @description Download competition leaderboards
#'
#'
#' @param competition the name of the competition
#' @param path a path to download the file to
#' @param quiet suppress verbose output (default is TRUE)
#' @return data frame
#'
#' @export
competition_leaderboard_download <- function(competition, path, quiet = TRUE) {

  res = kg$api$competition_leaderboard_download(
    competition = competition,
    path = path,
    quiet = quiet
  )[[1]]
  res = as.data.frame(do.call(rbind,res))
  res

}

#' @title Competitions list
#'
#'
#' @param group group to filter result to
#' @param category category to filter result to
#' @param sort_by how to sort the result, see valid_competition_sort_by for options
#' @param page the page to return (default is 1)
#' @param search a search term to use (default is empty string)
#' @return list of competitions
#' @export
competitions_list <- function(group = NULL, category = NULL, sort_by = NULL, page = 1, search = NULL) {

  kg$api$competitions_list(
    group = group,
    category = category,
    sort_by = sort_by,
    page = as.integer(page),
    search = search
  )

}



#' @title Competition submit
#'
#' @param file_name the competition metadata file
#' @param message the submission description
#' @param competition the competition name
#' @param quiet suppress verbose output (default is FALSE)
#' @return None
#' @export
competition_submit <- function(file_name, message, competition, quiet = FALSE) {

  kg$api$competition_submit(
    file_name = file_name,
    message = message,
    competition = competition,
    quiet = quiet
  )

}








