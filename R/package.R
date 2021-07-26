

fastai2 <- NULL
env <- new.env()

.onLoad <- function(libname, pkgname) {

  fastai2 <<- reticulate::import("fastai", delay_load = list(
    priority = 10,
    environment = "r-fastai"
  ))

  cran_ = !file.exists("C:/Users/ligges/AppData/Local/r-miniconda/envs/r-reticulate/python.exe")


  if(cran_) {

    if(reticulate::py_module_available('matplotlib')) {
      env[["matplot"]] <- reticulate::import('matplotlib')
      env[["matplot"]]$use('Agg')
      env[["warnings"]] <- reticulate::import('warnings')
      env[["warnings"]]$filterwarnings("ignore")
      env[['fix_fit']] <- fix_fit
      try(env[['fix_fit']](), TRUE)
      env[['bs_find']] <- bs_finder
      env[['bs_find']]()
    }


  }
}



#' Fix fit
#'
#'
#' @param disable_graph to remove dynamic plot, by default is FALSE
#' @return None
#' @export
fix_fit = function(disable_graph = FALSE) {
  fastaip <- reticulate::import('fastprogress')

  fastaip$progress_bar$fill = ''

  if(!disable_graph) {
    fastaip$fastprogress$WRITER_FN = function(value, ..., sep=' ', end='\n', flush = FALSE) {
      args = list(
        value, ...)

      text = unlist(strsplit(trimws(args[[1]]),' '))
      text = text[!text=='']
      lgl = grepl('epoch', text)

      # temp file
      nm = paste(tempdir(),'to_df.csv',sep = '/')

      # save column names // write to temp dir
      if(lgl[1]) {
        # remove old train from cache
        if(file.exists(nm))
          file.remove(nm)
        # write
        tmm = tempdir()
        tmp_name = paste(tmm,"output.txt",sep = '/')
        fileConn<-file(tmp_name)
        writeLines(text, fileConn)
        close(fileConn)
      }

      if(lgl[1]) {
        df <- data.frame(matrix(ncol = length(text), nrow = 0))
        colnames(df) <- text
        # add row for tidy output
        df[nrow(df) + 1,] = as.character(round(stats::runif(ncol(df)),3))
        df = knitr::kable(df, format = "pandoc")
        cat(df[1:2], sep="\n")

        if(interactive()) {
          try(dev.off(), TRUE)
        }


        set_theme = function() {
          ggplot2::theme_set(ggpubr::theme_pubr())
          utils::flush.console()
        }
        invisible(try(set_theme(), TRUE))

      } else {

        res=try(reticulate::py_run_string("
import sys
IN_COLAB = 'google.colab' in sys.modules
file = open('res.py', 'w')
file.write(str(IN_COLAB))
file.close()
"),silent=TRUE)

        if(res$IN_COLAB) {
          utils::flush.console()
        }

        ## restore from temp
        tmm = tempdir()
        tmp_name = paste(tmm,"output.txt",sep = '/')
        text2 = readLines(paste(tmm,"output.txt",sep = '/'))
        df <- data.frame(matrix(ncol = length(text2), nrow = 0))
        colnames(df) <- text2

        # add actual row
        silent_fun = function() {
          df[nrow(df) + 1,] = text
          df = knitr::kable(df, format = "pandoc")
          cat(df[3], sep="\n")
        }
        prnt = try(silent_fun(), TRUE)
        if(!inherits(prnt, 'try-error')) {
          # if !fail then repeat and collect data to temp dir
          df[nrow(df) + 1,] = text
          to_df = df
          to_df$time = NULL
          # if file is there, then read and row bind
          if(file.exists(nm)) {
            to_df_orig = read.csv(nm)
            to_df = rbind(to_df_orig, to_df)
            to_df$time = NULL
          }
          write.csv(to_df, nm, row.names = FALSE)
          #print(to_df)
          prnt
          # visualize but first make data frame numeric in case of character

          to_df = read.csv(nm)


          loss_names = grepl('loss', names(to_df))

          losses = cbind(to_df[1], to_df[loss_names])
          metrics_ = cbind(to_df[1], to_df[!names(to_df) %in% names(losses)])
          #print(losses)
          #print(metrics_)
          ## ggplot
          column_fun <- function(column_name, df, yaxis, colour) {

            lp <- ggplot2::ggplot(df, ggplot2::aes_string('epoch'))

            strings = column_name
            if(length(strings) > 1) {
              for (i in 1:length(strings)) {
                variable = ggplot2::sym(strings[i])
                lp <- lp + ggplot2::geom_line(ggplot2::aes(y = !!variable, colour = !!strings[i])) +
                  # add points
                  ggplot2::geom_point(ggplot2::aes(y = !!variable, colour = !!strings[i]))
              }
              lp <- lp +
                ggplot2::scale_x_continuous(breaks = seq(min(df)-1, max(df), 1)) +
                ggplot2::ylab(yaxis) + ggplot2::labs(colour = yaxis) + ggplot2::theme(legend.position="bottom",
                                                                                      legend.title=ggplot2::element_text(size=9),
                                                                                      legend.margin=ggplot2::margin(t = 0, unit='cm'),
                                                                                      axis.text=ggplot2::element_text(size=9),
                                                                                      axis.title=ggplot2::element_text(size=9,face="bold"))
            } else {
              variable <- ggplot2::sym(column_name)
              strings = column_name
              lp = lp + ggplot2::geom_line(ggplot2::aes(y = !!variable, colour = column_name)) +
                # add points
                ggplot2::geom_point(ggplot2::aes(y = !!variable, colour = column_name)) +
                ggplot2::scale_x_continuous(breaks = seq(min(df)-1, max(df), 1)) +
                ggplot2::ylab(yaxis) + ggplot2::labs(colour = yaxis) + ggplot2::theme(legend.position="bottom",
                                                                                      legend.title=ggplot2::element_text(size=9),
                                                                                      legend.margin=ggplot2::margin(t = 0, unit='cm'),
                                                                                      axis.text=ggplot2::element_text(size=9),
                                                                                      axis.title=ggplot2::element_text(size=9,face="bold"))
            }
            lp
          }


          result_fun = function() {
            if(nrow(to_df)>1) {
              if(ncol(metrics_)>1 & ncol(losses)>1) {
                p1 = column_fun(names(metrics_)[!names(metrics_) %in% 'epoch'], metrics_, 'Metrics', 'darkgreen')
                p2 = column_fun(names(losses)[!names(losses) %in% 'epoch'], losses, 'Loss', 'red')

                figure <- ggpubr::ggarrange(p2, p1,
                                            labels = c("", ""),
                                            ncol = 1, nrow = 2)
                print(figure)
              } else if (ncol(metrics_)>1 & ncol(losses)<=1) {
                p1 = column_fun(names(metrics_)[!names(metrics_) %in% 'epoch'], metrics_, 'Metrics', 'darkgreen')
                print(p1)

              } else if (ncol(metrics_)<=1 & ncol(losses)>1) {
                p2 = column_fun(names(losses)[!names(losses) %in% 'epoch'], losses, 'Loss', 'red')
                print(p2)

              } else {
                'None'
              }
            }
            paste('done plot')
          }

          if(interactive()){
            try(result_fun(), TRUE)
          }

        }
      }

    }
  } else {


    fastaip$fastprogress$WRITER_FN = function(value, ..., sep=' ', end='\n', flush = FALSE) {
      args = list(
        value, ...)

      text = unlist(strsplit(trimws(args[[1]]),' '))
      text = text[!text=='']
      lgl = grepl('epoch', text)

      # save column names // write to temp dir
      if(lgl[1]) {
        tmm = tempdir()
        tmp_name = paste(tmm,"output.txt",sep = '/')
        fileConn <- file(tmp_name)
        writeLines(text, fileConn)
        close(fileConn)
      }

      if(lgl[1]) {
        df <- data.frame(matrix(ncol = length(text), nrow = 0))
        colnames(df) <- text
        # add row for tidy output
        df[nrow(df) + 1,] = as.character(round(stats::runif(ncol(df)),3))
        df = knitr::kable(df, format = "pandoc")
        cat(df[1:2], sep="\n")
      } else {
        ## restore from temp
        tmm = tempdir()
        tmp_name = paste(tmm,"output.txt",sep = '/')
        text2 = readLines(paste(tmm,"output.txt",sep = '/'))
        df <- data.frame(matrix(ncol = length(text2), nrow = 0))
        colnames(df) <- text2
        # add actual row
        silent_fun = function() {
          df[nrow(df) + 1,] = text
          df = knitr::kable(df, format = "pandoc")
          cat(df[3], sep="\n")
        }
        try(silent_fun(), TRUE)
      }

    }


  }

}




