

#' @title Install fastai
#' @param version specify version
#' @param gpu installation of gpu
#' @param cuda_version if gpu true, then cuda version is required. By default it is 10.1
#' @param overwrite will install all the dependencies
#' @param extra_pkgs character vector of additional packages
#' @importFrom reticulate py_install
#' @param TPU official way to install Pytorch-XLA 1.7
#' @return None
#' @export
install_fastai <- function(version, gpu = FALSE, cuda_version = '10.1', overwrite = FALSE,
                           extra_pkgs = c('timm','fastinference[interp]'),
                           TPU = FALSE) {
  # extensions
  # 'blurr', 'icevision[all]', 'kaggle', 'transformers', git+https://github.com/tmabraham/UPIT.git
  # git+https://github.com/tcapelle/timeseries_fastai.git

  required_py_pkgs <- c('IPython', 'torch', 'torchvision', 'fastai',
                       'pydicom', 'kornia', 'cv2',
                       'skimage')


  if(length(extra_pkgs) > 0) {
    required_py_pkgs = c(required_py_pkgs, extra_pkgs)
  }

  required_py_pkgs = replace(required_py_pkgs, required_py_pkgs=="fastinference[interp]", "fastinference")

  res_ = list()
  for (i in 1:length(required_py_pkgs)) {
    result <- reticulate::py_module_available(required_py_pkgs[i])
    res_[[i]] <- result
  }
  res_ <- do.call(rbind, res_)[,1]
  which_pkgs <- which(res_ == FALSE)

  if(overwrite)
    required_py_pkgs = c(required_py_pkgs, extra_pkgs)
  else
    required_py_pkgs <- required_py_pkgs[which_pkgs]

  required_py_pkgs = replace(required_py_pkgs, required_py_pkgs=="cv2", "opencv-python")
  required_py_pkgs = replace(required_py_pkgs, required_py_pkgs=="skimage", "scikit-image")
  required_py_pkgs = replace(required_py_pkgs, required_py_pkgs=="shap", "shap==0.35.0")
  required_py_pkgs = replace(required_py_pkgs, required_py_pkgs=="fastinference", "fastinference[interp]")

  if(missing(version)) {
    required_py_pkgs = replace(required_py_pkgs, required_py_pkgs=="fastai", "fastai")
  } else {
    required_py_pkgs = replace(required_py_pkgs, required_py_pkgs=="fastai", paste("fastai",version,sep = "=="))
  }

  # linux
  cuda_linux = c('torch==1.7.1+cu92 torchvision==0.8.2+cu92 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html',
                 'torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html',
                 'torch torchvision torchaudio',
                 'torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html')
  linux_cpu = c('torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html')

  xla = "-U cloud-tpu-client==0.10 https://storage.googleapis.com/tpu-pytorch/wheels/torch_xla-1.7-cp36-cp36m-linux_x86_64.whl"

  # windows
  cuda_windows = c('Follow instructions at this URL: https://github.com/pytorch/pytorch#from-source',
                   'torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html',
                   'torch===1.7.1 torchvision===0.8.2 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html',
                   'torch===1.7.1+cu110 torchvision===0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html')
  cpu_windows = c('torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html')

  if('torch' %in% required_py_pkgs ) {
    torch_r = 'torch' %in% required_py_pkgs
    required_py_pkgs = required_py_pkgs[!required_py_pkgs %in% 'torch']
  } else {
    torch_r = character()
  }

  if('torchvision' %in% required_py_pkgs ) {
    torch_vision_r = 'torchvision' %in% required_py_pkgs
    required_py_pkgs = required_py_pkgs[!required_py_pkgs %in% 'torchvision']
  } else {
    torch_vision_r = character()
  }

  torch_r = c(torch_r, torch_vision_r)
  if(length(torch_r) > 0) {
    torch_r = all(torch_r == TRUE)
  } else {
    torch_r = FALSE
  }

  py_av = reticulate::py_available(TRUE)

  if (!length(required_py_pkgs) == 0) {

    if(py_av) {

      if (os() %in% 'linux' & !length(required_py_pkgs) == 0 & !TPU) {
        if(os() %in% 'linux' & gpu & cuda_version %in% '9.2' & torch_r & !length(required_py_pkgs) == 0) {
          py_install(packages = c(required_py_pkgs, cuda_linux[1]), pip = TRUE)

        } else if (os() %in% 'linux' & gpu & cuda_version %in% '10.1' & torch_r & !length(required_py_pkgs) == 0) {
          py_install(packages = c(required_py_pkgs, cuda_linux[2]), pip = TRUE)

        } else if (os() %in% 'linux' & gpu & cuda_version %in% '10.2' & torch_r & !length(required_py_pkgs) == 0) {
          py_install(packages = c(required_py_pkgs, cuda_linux[3]), pip = TRUE)

        } else if (os() %in% 'linux' & gpu & cuda_version %in% '11' & torch_r & !length(required_py_pkgs) == 0) {
          py_install(packages = c(required_py_pkgs, cuda_linux[4]), pip = TRUE)

        } else if(!gpu & torch_r & !length(required_py_pkgs) == 0) {
          py_install(packages = c(linux_cpu, required_py_pkgs), pip = TRUE)

        } else if (!torch_r & !length(required_py_pkgs) == 0) {
          py_install(packages = c(required_py_pkgs), pip = TRUE)

        } else {
          print('Fastai is installed!')
        }
      } else if (os() %in% 'linux' & length(required_py_pkgs) == 0) {
        print('Fastai is installed!')
      } else if (os() %in% 'linux' & TPU) {

        if(!missing(version) & os() %in% 'linux' & TPU)
          py_install(packages = c(cuda_linux[2], xla,paste("fastai",version,sep = '=='),'fastai_xla_extensions'), pip = TRUE)
        else if (missing(version) & os() %in% 'linux' & TPU)
          py_install(packages = c(cuda_linux[2], xla,"fastai",'fastai_xla_extensions'), pip = TRUE)

      }

      if (os() %in% 'windows' & !length(required_py_pkgs) == 0 & torch_r & !length(required_py_pkgs) == 0) {
        if(os() %in% 'windows' & gpu & cuda_version %in% '9.2') {
          print(cuda_windows[1])
        } else if (os() %in% 'windows' & gpu & cuda_version %in% '10.1' & torch_r & !length(required_py_pkgs) == 0) {
          py_install(packages = c(required_py_pkgs, cuda_windows[2]), pip = TRUE)

        } else if (os() %in% 'windows' & gpu & cuda_version %in% '10.2' & torch_r & !length(required_py_pkgs) == 0) {
          py_install(packages = c(required_py_pkgs, cuda_windows[3]), pip = TRUE)

        } else if (os() %in% 'windows' & gpu & cuda_version %in% '11' & torch_r & !length(required_py_pkgs) == 0) {
          py_install(packages = c(required_py_pkgs, cuda_windows[4]), pip = TRUE)

        } else if(!gpu & torch_r & !length(required_py_pkgs) == 0) {
          py_install(packages = c(cpu_windows, required_py_pkgs), pip = TRUE)

        } else if (!torch_r & !length(required_py_pkgs) == 0) {
          py_install(packages = c(required_py_pkgs), pip = TRUE)

        } else {
          print('Fastai is installed')
        }
      } else if (os() %in% 'windows' & length(required_py_pkgs) == 0){
        print('Fastai is installed!')
      }

      if (os() %in% 'mac' & !length(required_py_pkgs) == 0 & torch_r) {
        py_install(packages = c('torch torchvision torchaudio', required_py_pkgs), pip = TRUE)

      } else if (os() %in% 'mac' & !length(required_py_pkgs) == 0 & !torch_r){
        py_install(packages = c(required_py_pkgs), pip = TRUE)

      } else {
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


#' @title Fastai version
#'
#'
#'
#' @return None
#' @export
fastai_version = function() {
  fastai2$`__version__`
}





