#' @title competition_list_files
#'
#' @description list files for competition
#'
#'
#' @param competition competition the name of the competition
#'
#' @export
competition_list_files <- function(competition) {

  kg$api$competition_list_files(
    competition = competition
  )

}


#' @title competition_download_file
#'
#' @description download a competition file to a designated location, or use
#'
#' @details a default location Paramters
#' =========
#' competition: the name of the competition
#' file_name: the configuration file name
#' path: a path to download the file to
#' force: force the download if the file already exists (default FALSE)
#' quiet: suppress verbose output (default is FALSE)
#'
#' @param competition competition
#' @param file_name file_name
#' @param path path
#' @param force force
#' @param quiet quiet
#'
#' @section competition: the name of the competition:
#' file_name: the configuration file name path: a path to download the file to force: force the download if the file already exists (default FALSE) quiet: suppress verbose output (default is FALSE)
#'
#' @section file_name: the configuration file name:
#' path: a path to download the file to force: force the download if the file already exists (default FALSE) quiet: suppress verbose output (default is FALSE)
#'
#' @section path: a path to download the file to:
#' force: force the download if the file already exists (default FALSE) quiet: suppress verbose output (default is FALSE)
#'
#' @section force: force the download if the file already exists (default False):
#' quiet: suppress verbose output (default is FALSE)
#'
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

#' @title competition_download_files
#'
#' @description downloads all competition files.
#'
#' @details Parameters
#' =========
#' competition: the name of the competition
#' path: a path to download the file to
#' force: force the download if the file already exists (default FALSE)
#' quiet: suppress verbose output (default is TRUE)
#'
#' @param competition competition
#' @param path path
#' @param force force
#' @param quiet quiet
#'
#' @section competition: the name of the competition:
#' path: a path to download the file to force: force the download if the file already exists (default FALSE) quiet: suppress verbose output (default is TRUE)
#'
#' @section path: a path to download the file to:
#' force: force the download if the file already exists (default FALSE) quiet: suppress verbose output (default is TRUE)
#'
#' @section force: force the download if the file already exists (default False):
#' quiet: suppress verbose output (default is TRUE)
#'
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




#' @title competition_leaderboard_download
#'
#' @description Download competition leaderboards
#'
#' @details Parameters
#' =========
#' competition: the name of the competition
#' path: a path to download the file to
#' quiet: suppress verbose output (default is TRUE)
#'
#' @param competition competition
#' @param path path
#' @param quiet quiet
#'
#' @section competition: the name of the competition:
#' path: a path to download the file to quiet: suppress verbose output (default is TRUE)
#'
#' @section path: a path to download the file to:
#' quiet: suppress verbose output (default is TRUE)
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

#' @title competitions_list
#'
#' @description make call to list competitions, format the response, and return
#'
#' @details a list of Competition instances Parameters
#' ========== page: the page to return (default is 1)
#' search: a search term to use (default is empty string)
#' sort_by: how to sort the result, see valid_competition_sort_by for options
#' category: category to filter result to
#' group: group to filter result to
#'
#' @param group group
#' @param category category
#' @param sort_by sort_by
#' @param page page
#' @param search search
#'
#' @section page: the page to return (default is 1):
#' search: a search term to use (default is empty string) sort_by: how to sort the result, see valid_competition_sort_by for options category: category to filter result to group: group to filter result to
#'
#' @section search: a search term to use (default is empty string):
#' sort_by: how to sort the result, see valid_competition_sort_by for options category: category to filter result to group: group to filter result to
#'
#' @section sort_by: how to sort the result, see valid_competition_sort_by for options:
#' category: category to filter result to group: group to filter result to
#'
#' @section category: category to filter result to:
#' group: group to filter result to
#'
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



#' @title competition_submit
#'
#' @description submit a competition!
#'
#' @details Parameters
#' ==========
#' file_name: the competition metadata file
#' message: the submission description
#' competition: the competition name
#' quiet: suppress verbose output (default is FALSE)
#'
#' @param file_name file_name
#' @param message message
#' @param competition competition
#' @param quiet quiet
#'
#' @section file_name: the competition metadata file:
#' message: the submission description competition: the competition name quiet: suppress verbose output (default is FALSE)
#'
#' @section message: the submission description:
#' competition: the competition name quiet: suppress verbose output (default is FALSE)
#'
#' @section competition: the competition name:
#' quiet: suppress verbose output (default is FALSE)
#'
#' @export
competition_submit <- function(file_name, message, competition, quiet = FALSE) {

  kg$api$competition_submit(
    file_name = file_name,
    message = message,
    competition = competition,
    quiet = quiet
  )

}








