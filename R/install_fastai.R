

#' @title Install fastai
#' @param version specify version
#' @param gpu installation of gpu
#' @param cuda_version if gpu true, then cuda version is required. By default it is 10.1
#' @param overwrite will install all the dependencies
#' @param extra_pkgs character vector of additional packages
#' @importFrom reticulate py_install
#' @return None
#' @export
install_fastai <- function(version, gpu = FALSE, cuda_version = '10.1', overwrite = FALSE,
                           extra_pkgs = c('kaggle', 'transformers', 'pytorch_lightning', 'timm',
                                          'catalyst', 'ignite', 'tensorboard', 'fastinference', 'shap')) {

  required_py_pkgs <- c('IPython', 'torch', 'torchvision', 'fastai',
                       'pydicom', 'kornia', 'cv2',
                       'skimage')
  # if git is available
  git = try(suppressWarnings(system('which git', intern = TRUE)), TRUE)

  # audio, time-series, cycle-GAN, transformers integration==blurr
  git_pkgs = c('fastaudio', 'timeseries_fastai', 'blurr', 'upit')

  if(length(extra_pkgs) > 0) {
    required_py_pkgs = c(required_py_pkgs, extra_pkgs, git_pkgs)
  }

  res_ = list()
  for (i in 1:length(required_py_pkgs)) {
    result <- reticulate::py_module_available(required_py_pkgs[i])
    res_[[i]] <- result
  }
  res_ <- do.call(rbind, res_)[,1]
  which_pkgs <- which(res_ == FALSE)

  if(overwrite)
    required_py_pkgs = c(required_py_pkgs, extra_pkgs, git_pkgs)
  else
    required_py_pkgs <- required_py_pkgs[which_pkgs]


  #required_py_pkgs <- required_py_pkgs[!required_py_pkgs %in% c('torch','torchvision')]

  required_py_pkgs = replace(required_py_pkgs, required_py_pkgs=="cv2", "opencv-python")
  required_py_pkgs = replace(required_py_pkgs, required_py_pkgs=="skimage", "scikit-image")
  required_py_pkgs = replace(required_py_pkgs, required_py_pkgs=="pytorch_lightning", "pytorch-lightning")
  required_py_pkgs = replace(required_py_pkgs, required_py_pkgs=="ignite", "pytorch-ignite")
  #required_py_pkgs = replace(required_py_pkgs, required_py_pkgs=="torchaudio", "torchaudio==0.6.0")
  required_py_pkgs = replace(required_py_pkgs, required_py_pkgs=="shap", "shap==0.35.0")

  # git pkgs
  required_py_pkgs = replace(required_py_pkgs, required_py_pkgs=="fastaudio", "git+https://github.com/fastaudio/fastaudio.git")
  required_py_pkgs = replace(required_py_pkgs, required_py_pkgs=="timeseries_fastai", "git+https://github.com/tcapelle/timeseries_fastai.git")
  required_py_pkgs = replace(required_py_pkgs, required_py_pkgs=="blurr", "git+https://github.com/ohmeow/blurr.git")
  required_py_pkgs = replace(required_py_pkgs, required_py_pkgs=="upit", "git+https://github.com/tmabraham/UPIT.git")

  if(missing(version)) {
    required_py_pkgs = replace(required_py_pkgs, required_py_pkgs=="fastai", "fastai")
  } else {
    required_py_pkgs = replace(required_py_pkgs, required_py_pkgs=="fastai", paste("fastai",version,sep = "=="))
  }

  # get os
  os = switch(Sys.info()[['sysname']],
              Windows= 'windows',
              Linux  = 'linux',
              Darwin = 'mac')

  # linux
  cuda_linux = c('torch==1.7.0+cu92 torchvision==0.8.1+cu92 torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html',
                 'torch==1.7.0+cu101 torchvision==0.8.1+cu101 torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html',
                 'torch torchvision torchaudio',
                 'torch==1.7.0+cu110 torchvision==0.8.1+cu110 torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html')
  linux_cpu = c('torch==1.7.0+cpu torchvision==0.8.1+cpu torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html')

  # windows
  cuda_windows = c('Follow instructions at this URL: https://github.com/pytorch/pytorch#from-source',
                   'torch==1.7.0+cu101 torchvision==0.8.1+cu101 torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html',
                   'torch===1.7.0 torchvision===0.8.1 torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html',
                   'torch===1.7.0+cu110 torchvision===0.8.1+cu110 torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html')
  cpu_windows = c('torch==1.7.0+cpu torchvision==0.8.1+cpu torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html')

  if('torch' %in% required_py_pkgs ) {
    required_py_pkgs = required_py_pkgs[!required_py_pkgs %in% 'torch']
  }

  if('torchvision' %in% required_py_pkgs ) {
    required_py_pkgs = required_py_pkgs[!required_py_pkgs %in% 'torchvision']
  }


  py_av = reticulate::py_available(TRUE)

  if (!length(required_py_pkgs) == 0) {

    if(py_av) {

      if (os %in% 'linux' & !length(required_py_pkgs) == 0) {
        if(os %in% 'linux' & gpu & cuda_version %in% '9.2') {
          py_install(packages = c(required_py_pkgs, cuda_linux[1]), pip = TRUE)
          py_install('fastinference[interp]', pip = TRUE)
        } else if (os %in% 'linux' & gpu & cuda_version %in% '10.1') {
          py_install(packages = c(required_py_pkgs, cuda_linux[2]), pip = TRUE)
          py_install('fastinference[interp]', pip = TRUE)
        } else if (os %in% 'linux' & gpu & cuda_version %in% '10.2') {
          py_install(packages = c(required_py_pkgs, cuda_linux[3]), pip = TRUE)
          py_install('fastinference[interp]', pip = TRUE)
        } else if (os %in% 'linux' & gpu & cuda_version %in% '11') {
          py_install(packages = c(required_py_pkgs, cuda_linux[4]), pip = TRUE)
          py_install('fastinference[interp]', pip = TRUE)
        } else {
          py_install(packages = c(linux_cpu, required_py_pkgs), pip = TRUE)
          py_install('fastinference[interp]', pip = TRUE)
        }
      } else if (os %in% 'linux' & length(required_py_pkgs) == 0) {
        print('Fastai is installed!')
      }

      if (os %in% 'windows' & !length(required_py_pkgs) == 0) {
        if(os %in% 'windows' & gpu & cuda_version %in% '9.2') {
          print(cuda_windows[1])
        } else if (os %in% 'windows' & gpu & cuda_version %in% '10.1') {
          py_install(packages = c(required_py_pkgs, cuda_windows[2]), pip = TRUE)
          py_install('fastinference[interp]', pip = TRUE)
        } else if (os %in% 'windows' & gpu & cuda_version %in% '10.2') {
          py_install(packages = c(required_py_pkgs, cuda_windows[3]), pip = TRUE)
          py_install('fastinference[interp]', pip = TRUE)
        } else if (os %in% 'windows' & gpu & cuda_version %in% '11') {
          py_install(packages = c(required_py_pkgs, cuda_windows[4]), pip = TRUE)
          py_install('fastinference[interp]', pip = TRUE)
        } else {
          py_install(packages = c(cpu_windows, required_py_pkgs), pip = TRUE)
          py_install('fastinference[interp]', pip = TRUE)
        }
      } else if (os %in% 'windows' & length(required_py_pkgs) == 0){
        print('Fastai is installed!')
      }

      if (os %in% 'mac' & !length(required_py_pkgs) == 0) {
        py_install(packages = c('torch torchvision torchaudio', required_py_pkgs), pip = TRUE)
        py_install('fastinference[interp]', pip = TRUE)
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
