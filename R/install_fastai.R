

#' @title Install fastai
#' @param version specify version
#' @param gpu installation of gpu
#' @param cuda_version if gpu true, then cuda version is required. By default it is 10.1
#' @param overwrite will install all the dependencies
#' @importFrom reticulate py_install
#' @export
install_fastai <- function(version, gpu = FALSE, cuda_version = '10.1', overwrite = FALSE) {
  invisible(reticulate::py_config())

  required_py_pkgs <- c('IPython', 'torch', 'torchvision', 'fastai',
                       'pydicom', 'kornia', 'cv2',
                       'skimage')

  res_ = list()
  for (i in 1:length(required_py_pkgs)) {
    result <- reticulate::py_module_available(required_py_pkgs[i])
    res_[[i]] <- result
  }
  res_ <- do.call(rbind, res_)[,1]
  which_pkgs <- which(res_ == FALSE)

  if(overwrite)
    required_py_pkgs
  else
    required_py_pkgs <- required_py_pkgs[which_pkgs]

  #required_py_pkgs <- required_py_pkgs[!required_py_pkgs %in% c('torch','torchvision')]

  required_py_pkgs = replace(required_py_pkgs, required_py_pkgs=="cv2", "opencv-python")
  required_py_pkgs = replace(required_py_pkgs, required_py_pkgs=="skimage", "scikit-image")
  required_py_pkgs = replace(required_py_pkgs, required_py_pkgs=="fastai", paste("fastai",version,sep = "=="))

  # get os
  os = switch(Sys.info()[['sysname']],
              Windows= 'windows',
              Linux  = 'linux',
              Darwin = 'mac')

  # linux
  cuda_linux = c('torch==1.6.0+cu92 torchvision==0.7.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html',
                 'torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html',
                 'torch torchvision')
  cpu_cpu = c('torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html')

  # windows
  cuda_windows = c('Follow instructions at this URL: https://github.com/pytorch/pytorch#from-source',
                   'torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html',
                   'torch===1.6.0 torchvision===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html')
  cpu_windows = c('torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html')


  if (!length(required_py_pkgs) == 0) {

    if(!reticulate::virtualenv_list() == 0) {

      if (os %in% 'linux' & !length(required_py_pkgs) == 0) {
        if(os %in% 'linux' & gpu & cuda_version %in% '9.2') {
          py_install(packages = c(required_py_pkgs, cuda_linux[1]), pip = TRUE)
        } else if (os %in% 'linux' & gpu & cuda_version %in% '10.1') {
          py_install(packages = c(required_py_pkgs, cuda_linux[2]), pip = TRUE)
        } else if (os %in% 'linux' & gpu & cuda_version %in% '10.2') {
          py_install(packages = c(required_py_pkgs, cuda_linux[3]), pip = TRUE)
        } else {
          py_install(packages = c(required_py_pkgs), pip = TRUE)
        }
      } else if (os %in% 'linux' & length(required_py_pkgs) == 0) {
        print('Fastai is installed!')
      }

      if (os %in% 'windows' & !length(required_py_pkgs) == 0) {
        if(os %in% 'windows' & gpu & cuda_version %in% '9.2') {
          print(cuda_windows[1])
        } else if (os %in% 'windows' & gpu & cuda_version %in% '10.1') {
          py_install(packages = c(required_py_pkgs, cuda_windows[2]), pip = TRUE)
        } else if (os %in% 'windows' & gpu & cuda_version %in% '10.2') {
          py_install(packages = c(required_py_pkgs, cuda_windows[3]), pip = TRUE)
        } else {
          py_install(packages = c(required_py_pkgs), pip = TRUE)
        }
      } else if (os %in% 'windows' & length(required_py_pkgs) == 0){
        print('Fastai is installed!')
      }

      if (os %in% 'mac' & !length(required_py_pkgs) == 0) {
        py_install(packages = c(required_py_pkgs, 'torch torchvision'), pip = TRUE)
      } else if (os %in% 'mac' & length(required_py_pkgs) == 0){
        print('Fastai is installed!')
      }

    } else {
      stop(
        c('Try to install miniconda and activate environment:\n1) reticulate::install_miniconda()\n2) reticulate::py_config()'),
        call. = FALSE)
    }
  } else {
    print('Fastai is installed!')
  }

}




