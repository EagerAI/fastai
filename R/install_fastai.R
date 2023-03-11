

#' @title Install fastai
#' @param version specify version
#' @param gpu installation of gpu
#' @param cuda_version if gpu true, then cuda version is required. By default it is 11.6
#' @param overwrite will install all the dependencies
#' @param extra_pkgs character vector of additional packages
#' @importFrom reticulate py_install
#' @param TPU official way to install Pytorch-XLA 1.13
#' @return None
#' @export
install_fastai <- function(version, gpu = FALSE, cuda_version = '11.6', overwrite = FALSE,
                           extra_pkgs = c('timm','fastinference[interp]'),
                           TPU = FALSE) {
  # extensions
  # 'blurr', 'icevision[all]', 'kaggle', 'transformers', git+https://github.com/tmabraham/UPIT.git
  # git+https://github.com/tcapelle/timeseries_fastai.git
  # 'fastinference[interp]'

  required_py_pkgs <- c('IPython', 'torch', 'torchvision', 'fastai',
                       'pydicom', 'kornia', 'cv2',
                       'skimage')

  if(os()=='linux') {
    required_py_pkgs <- append(required_py_pkgs,'ohmeow-blurr')
  }


  if(length(extra_pkgs) > 0) {
    required_py_pkgs = c(required_py_pkgs, extra_pkgs)
  }

  required_py_pkgs = replace(required_py_pkgs, required_py_pkgs=="fastinference[interp]", "fastinference")
  required_py_pkgs = replace(required_py_pkgs, required_py_pkgs=="ohmeow-blurr", "blurr")

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
  required_py_pkgs = replace(required_py_pkgs, required_py_pkgs=="blurr", "ohmeow-blurr")

  if(missing(version)) {
    required_py_pkgs = replace(required_py_pkgs, required_py_pkgs=="fastai", "fastai")
  } else if(!missing(version) & grep('git', version)==1) {
    required_py_pkgs = replace(required_py_pkgs, required_py_pkgs=="fastai", version)
  } else {
    required_py_pkgs = replace(required_py_pkgs, required_py_pkgs=="fastai", paste("fastai",version,sep = "=="))
  }

  # linux
  cuda_linux = c('torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116',
                 'torch torchvision torchaudio')
  linux_cpu = c('torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu')

  xla = "cloud-tpu-client==0.10 torch==1.13.0 https://storage.googleapis.com/tpu-pytorch/wheels/colab/torch_xla-1.13-cp38-cp38-linux_x86_64.whl tensorboard-plugin-profile"

  # windows
  cuda_windows = c('torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116',
                   'torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117')
  cpu_windows = c('torch torchvision torchaudio')

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

  fastai_py_install = function(pkgs) {
    system(paste(reticulate::py_discover_config()[['python']],'-m pip install',
                 paste(pkgs,collapse = ' ')
                 ))
  }

  if (!length(required_py_pkgs) == 0) {

    if(py_av) {

      if (os() %in% 'linux' & !length(required_py_pkgs) == 0 & !TPU) {
        if (os() %in% 'linux' & gpu & cuda_version %in% '11.6' & torch_r & !length(required_py_pkgs) == 0) {
           fastai_py_install( c(required_py_pkgs, cuda_linux[1]) )

        } else if (os() %in% 'linux' & gpu & cuda_version %in% '11.7' & torch_r & !length(required_py_pkgs) == 0) {
          fastai_py_install(c(required_py_pkgs, cuda_linux[2]) )

        } else if(!gpu & torch_r & !length(required_py_pkgs) == 0) {
          fastai_py_install( c(linux_cpu, required_py_pkgs))

        } else if (!torch_r & !length(required_py_pkgs) == 0) {
          fastai_py_install( required_py_pkgs)

        } else {
          print('Fastai is installed!')
        }
      } else if (os() %in% 'linux' & length(required_py_pkgs) == 0) {
        print('Fastai is installed!')
      } else if (os() %in% 'linux' & TPU) {

        if(!missing(version) & os() %in% 'linux' & TPU)
           fastai_py_install( c(cuda_linux[2], xla,paste("fastai",version,sep = '=='),'fastai_xla_extensions') )
        else if (missing(version) & os() %in% 'linux' & TPU)
          fastai_py_install( c(cuda_linux[2], xla,"fastai",'fastai_xla_extensions') )

      }

      if (os() %in% 'windows' & !length(required_py_pkgs) == 0 & torch_r & !length(required_py_pkgs) == 0) {
        if (os() %in% 'windows' & gpu & cuda_version %in% '11.6' & torch_r & !length(required_py_pkgs) == 0) {
          fastai_py_install( c(required_py_pkgs, cuda_windows[1]) )

        } else if (os() %in% 'windows' & gpu & cuda_version %in% '11.7' & torch_r & !length(required_py_pkgs) == 0) {
         fastai_py_install( c(required_py_pkgs, cuda_windows[2]) )

        } else if(!gpu & torch_r & !length(required_py_pkgs) == 0) {
          fastai_py_install( c(cpu_windows, required_py_pkgs) )

        } else if (!torch_r & !length(required_py_pkgs) == 0) {
          fastai_py_install( required_py_pkgs )

        } else {
          print('Fastai is installed')
        }
      } else if (os() %in% 'windows' & length(required_py_pkgs) == 0){
        print('Fastai is installed!')
      }

      if (os() %in% 'mac' & !length(required_py_pkgs) == 0 & torch_r) {
       fastai_py_install( c('torch torchvision torchaudio', required_py_pkgs ) )

      } else if (os() %in% 'mac' & !length(required_py_pkgs) == 0 & !torch_r){
      fastai_py_install( required_py_pkgs )

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
#' @return None
#' @export
fastai_version = function() {
  fastai2$`__version__`
}





