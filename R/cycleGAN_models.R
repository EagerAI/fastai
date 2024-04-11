


#' @title ConvT_norm_relu
#' @param ch_in input
#' @param ch_out output
#' @param norm_layer normalziation layer
#' @param ks kernel size
#' @param stride stride size
#' @param bias bias true or not
#' @return None
#' @export
convT_norm_relu <- function(ch_in, ch_out, norm_layer, ks = 3, stride = 2, bias = TRUE) {

  args = list(
    ch_in = as.integer(ch_in),
    ch_out = as.integer(ch_out),
    norm_layer = norm_layer,
    ks = as.integer(ks),
    stride = as.integer(stride),
    bias = bias
  )

  do.call(upit()$models$cyclegan$convT_norm_relu, args)

}


#' @title Pad_conv_norm_relu
#' @param ch_in input
#' @param ch_out output
#' @param pad_mode padding mode
#' @param norm_layer normalization layer
#' @param ks kernel size
#' @param bias bias
#' @param pad padding
#' @param stride stride
#' @param activ activation
#' @param init initializer
#' @param init_gain init gain
#' @return None
#' @export
pad_conv_norm_relu <- function(ch_in, ch_out, pad_mode, norm_layer, ks = 3,
                               bias = TRUE, pad = 1, stride = 1, activ = TRUE,
                               init = nn()$init$kaiming_normal_, init_gain = 0.02) {

  args <- list(
    ch_in = as.integer(ch_in),
    ch_out = as.integer(ch_out),
    pad_mode = pad_mode,
    norm_layer = norm_layer,
    ks = as.integer(ks),
    bias = bias,
    pad = as.integer(pad),
    stride = as.integer(stride),
    activ = activ,
    init = init,
    init_gain = init_gain
  )

  do.call(upit()$models$cyclegan$pad_conv_norm_relu, args)

}


#' @title ResnetBlock
#'
#' @description nn()$Module for the ResNet Block
#'
#'
#' @param dim dimension
#' @param pad_mode padding mode
#' @param norm_layer normalization layer
#' @param dropout dropout rate
#' @param bias bias or not
#' @return None
#' @export
ResnetBlock <- function(dim, pad_mode = "reflection", norm_layer = NULL, dropout = 0.0, bias = TRUE) {


  if(missing( dim) ) {
    upit()$models$cyclegan$ResnetBlock
  } else {
    args <- list(
      dim = as.integer(dim),
      pad_mode = pad_mode,
      norm_layer = norm_layer,
      dropout = dropout,
      bias = bias
    )

    do.call(upit()$models$cyclegan$ResnetBlock, args)
  }
}



#' @title Resnet_generator
#' @param ch_in input
#' @param ch_out output
#' @param n_ftrs filter
#' @param norm_layer normalziation layer
#' @param dropout dropout rate
#' @param n_blocks number of blocks
#' @param pad_mode paddoing mode
#' @return None
#' @export
resnet_generator <- function(ch_in, ch_out, n_ftrs = 64, norm_layer = NULL,
                             dropout = 0.0, n_blocks = 9, pad_mode = "reflection") {

  args <- list(
    ch_in = as.integer(ch_in),
    ch_out = as.integer(ch_out),
    n_ftrs = as.integer(n_ftrs),
    norm_layer = norm_layer,
    dropout = dropout,
    n_blocks = as.integer(n_blocks),
    pad_mode = pad_mode
  )

  if(is.null(args$norm_layer))
    args$norm_layer <- NULL

  do.call(upit()$models$cyclegan$resnet_generator, args)

}



#' @title Conv_norm_lr
#' @param ch_in input
#' @param ch_out output
#' @param norm_layer normalziation layer
#' @param ks kernel size
#' @param bias bias
#' @param pad pad
#' @param stride stride
#' @param activ activation
#' @param slope slope
#' @param init inititializer
#' @param init_gain initializer gain
#' @return None
#' @export
conv_norm_lr <- function(ch_in, ch_out, norm_layer = NULL, ks = 3, bias = TRUE,
                         pad = 1, stride = 1, activ = TRUE, slope = 0.2,
                         init = nn()$init$normal_, init_gain = 0.02) {

  args <- list(
    ch_in = as.integer(ch_in),
    ch_out = as.integer(ch_out),
    norm_layer = norm_layer,
    ks = as.integer(ks),
    bias = bias,
    pad = as.integer(pad),
    stride = as.integer(stride),
    activ = activ,
    slope = slope,
    init = init,
    init_gain = init_gain
  )

  if(is.null(args$norm_layer))
    args$norm_layer <- NULL

  do.call(upit()$models$cyclegan$conv_norm_lr, args)

}


#' @title Discriminator
#' @param ch_in input
#' @param n_ftrs number of filters
#' @param n_layers number of layers
#' @param norm_layer normalization layer
#' @param sigmoid apply sigmoid  function or not
#'
#' @export
discriminator <- function(ch_in, n_ftrs = 64, n_layers = 3, norm_layer = NULL, sigmoid = FALSE) {

  args = list(
    ch_in = as.integer(ch_in),
    n_ftrs = as.integer(n_ftrs),
    n_layers = as.integer(n_layers),
    norm_layer = norm_layer,
    sigmoid = sigmoid
  )

  if(is.null(args$norm_layer))
    args$norm_layer <- NULL

  do.call(upit()$models$cyclegan$discriminator,args)

}


#' @title CycleGAN
#'
#' @description CycleGAN model.
#'
#' @details When called, takes in input batch of real images from both domains and outputs fake images for the opposite domains (with the generators).
#' Also outputs identity images after passing the images into generators that outputs its domain type (needed for identity loss). Attributes: `G_A` (`nn.Module`): takes real input B and generates fake input A `G_B` (`nn.Module`): takes real input A and generates fake input B `D_A` (`nn.Module`): trained to make the difference between real input A and fake input A `D_B` (`nn.Module`): trained to make the difference between real input B and fake input B
#'
#' @param ch_in input
#' @param ch_out output
#' @param n_features number of features
#' @param disc_layers discriminator layers
#' @param gen_blocks generator blocks
#' @param lsgan ls gan
#' @param drop dropout rate
#' @param norm_layer normalziation layer
#' @return None
#' @export
CycleGAN <- function(ch_in = 3, ch_out = 3, n_features = 64, disc_layers = 3,
                     gen_blocks = 9, lsgan = TRUE, drop = 0.0, norm_layer = NULL) {

  args <- list(
    ch_in = as.integer(ch_in),
    ch_out = as.integer(ch_out),
    n_features = as.integer(n_features),
    disc_layers = as.integer(disc_layers),
    gen_blocks = as.integer(gen_blocks),
    lsgan = lsgan,
    drop = drop,
    norm_layer = norm_layer
  )

  if(is.null(args$norm_layer))
    args$norm_layer <- NULL

  do.call(upit()$models$cyclegan$CycleGAN, args)

}


#' @title RandPair
#'
#' @description a random image from domain B, resulting in a random pair of images from domain A and B.
#'
#'
#' @param itemsB a random image from domain B
#' @return None
#' @export
RandPair <- function(itemsB) {

  upit()$data$unpaired$RandPair(
    itemsB = itemsB
  )

}


#' @title Get dls
#'
#' @description Given image files from two domains (`pathA`, `pathB`), create `DataLoaders` object.
#'
#' @details Loading and randomly cropped sizes of `load_size` and `crop_size` are set to defaults of 512 and 256.
#' Batch size is specified by `bs` (default=4).
#'
#' @param pathA path A (from domain)
#' @param pathB path B (to domain)
#' @param num_A subset of A data
#' @param num_B subset of B data
#' @param load_size load size
#' @param crop_size crop size
#' @param bs bathc size
#' @param num_workers number of workers
#' @return None
#' @export
get_dls <- function(pathA, pathB, num_A = NULL, num_B = NULL,
                    load_size = 512, crop_size = 256, bs = 4,
                    num_workers = 2) {

 args <- list(
    pathA = pathA,
    pathB = pathB,
    num_A = num_A,
    num_B = num_B,
    load_size = as.integer(load_size),
    crop_size = as.integer(crop_size),
    bs = as.integer(bs),
    num_workers = as.integer(num_workers)
  )

 if(!is.null(args[['num_A']]))
   args[['num_A']] = as.integer(args[['num_A']])
 else
   args[['num_A']] <- NULL

 if(!is.null(args[['num_B']]))
   args[['num_B']] = as.integer(args[['num_B']])
 else
   args[['num_B']] <- NULL

 do.call(upit()$data$unpaired$get_dls, args)

}



#' @title CycleGANLoss
#'
#' @description CycleGAN loss function. The individual loss terms are also atrributes
#' of this class that are accessed by fastai for recording during training.
#'
#' @details Attributes: `self.cgan` (`nn.Module`): The CycleGAN model.
#' `self.l_A` (`float`): lambda_A, weight of domain A losses.
#' `self.l_B` (`float`): lambda_B, weight of domain B losses.
#' `self.l_idt` (`float`): lambda_idt, weight of identity lossees.
#' `self.crit` (`AdaptiveLoss`): The adversarial loss function
#' (either a BCE or MSE loss depending on `lsgan` argument)
#' `self.real_A` and `self.real_B` (`fastai.torch_core.TensorImage`): Real images from domain A and B.
#' `self.id_loss_A` (`torch.FloatTensor`): The identity loss for domain A calculated
#' in the forward function `self.id_loss_B` (`torch.FloatTensor`): The identity loss for domain B calculated
#' in the forward function `self.gen_loss` (`torch.FloatTensor`): The generator loss calculated
#' in the forward function `self.cyc_loss` (`torch.FloatTensor`): The cyclic loss calculated
#' in the forward function
#'
#' @param cgan The CycleGAN model.
#' @param l_A lambda_A, weight of domain A losses. (default=10)
#' @param l_B lambda_B, weight of domain B losses. (default=10)
#' @param l_idt lambda_idt, weight of identity lossees. (default=0.5)
#' @param lsgan Whether or not to use LSGAN objective (default=True)
#'
#' @export
CycleGANLoss <- function(cgan, l_A = 10.0, l_B = 10, l_idt = 0.5, lsgan = TRUE) {

  upit()$train$cyclegan$CycleGANLoss(
    cgan = cgan,
    l_A = l_A,
    l_B = l_B,
    l_idt = l_idt,
    lsgan = lsgan
  )

}


#' CycleGANTrainer
#'
#' @description Learner Callback for training a CycleGAN model.
#' @param ... parameters to pass
#'
#' @return None
CycleGANTrainer = function(...) {
  args = list(...)
  do.call(upit()$train$cyclegan$CycleGANTrainer, args)
}

#' @title ShowCycleGANImgsCallback
#'
#' @description Update the progress bar with input and prediction images
#'
#'
#' @param imgA img from A domain
#' @param imgB img from B domain
#' @param show_img_interval show image interval
#' @return None
#' @export
ShowCycleGANImgsCallback <- function(imgA = FALSE, imgB = TRUE, show_img_interval = 10) {

  upit()$train$cyclegan$ShowCycleGANImgsCallback(
    imgA = imgA,
    imgB = imgB,
    show_img_interval = as.integer(show_img_interval)
  )

}



#' @title Combined_flat_anneal
#'
#' @description Create a schedule with constant learning rate `start_lr` for `pct`
#' proportion of the training, and a `curve_type` learning rate (till `end_lr`) for
#' remaining portion of training.
#'
#'
#' @param pct Proportion of training with a constant learning rate.
#' @param start_lr Desired starting learning rate, used for beginnning pct of training.
#' @param end_lr  Desired end learning rate, training will conclude at this learning rate.
#' @param curve_type Curve type for learning rate annealing. Options are 'linear', 'cosine', and 'exponential'.
#'
#' @export
combined_flat_anneal <- function(pct, start_lr, end_lr = 0, curve_type = "linear") {

  upit()$train$cyclegan$combined_flat_anneal(
    pct = pct,
    start_lr = start_lr,
    end_lr = end_lr,
    curve_type = curve_type
  )

}


#' @title Cycle_learner
#'
#' @description Initialize and return a `Learner` object with the data in `dls`, CycleGAN model `m`, optimizer function `opt_func`, metrics `metrics`,
#'
#' @details and callbacks `cbs`. Additionally, if `show_imgs` is TRUE, it will show intermediate predictions during training. It will show domain
#' B-to-A predictions if `imgA` is TRUE and/or domain A-to-B predictions if `imgB` is TRUE. Additionally, it will show images every
#' `show_img_interval` epochs. ` Other `Learner` arguments can be passed as well.
#'
#' @param dls dataloader
#' @param m CycleGAN model
#' @param opt_func optimizer
#' @param show_imgs show images
#' @param imgA image a (from)
#' @param imgB image B (to)
#' @param show_img_interval show images interval rafe
#' @param ... additional arguments
#' @return None
#' @export
cycle_learner <- function(dls, m, opt_func = Adam(), show_imgs = TRUE,
                          imgA = TRUE, imgB = TRUE, show_img_interval = 10,
                          ...) {

  args <- list(
    dls = dls,
    m = m,
    opt_func = opt_func,
    show_imgs = show_imgs,
    imgA = imgA,
    imgB = imgB,
    show_img_interval = as.integer( show_img_interval),
    ...
  )

  do.call(upit()$train$cyclegan$cycle_learner, args)

}



#' @title HORSE_2_ZEBRA dataset
#'
#' @param filename the name of the file
#' @param unzip logical, whether to unzip the '.zip' file
#'
#' @description download HORSE_2_ZEBRA dataset
#' @return None
#' @export
URLs_HORSE_2_ZEBRA <- function(filename = 'horse2zebra', unzip = TRUE) {

  download.file('https://efrosgans.eecs.berkeley.edu/cyclegan/datasets/horse2zebra.zip',
                destfile = paste(filename,'.zip',sep = ''))

  if(unzip)
    unzip(paste(filename,'.zip',sep = ''))

}



#' @title FolderDataset
#'
#' @description A PyTorch Dataset class that can be created from a folder `path` of images, for the sole purpose of inference. Optional `transforms`
#'
#' @details can be provided. Attributes: `self.files`: A list of the filenames in the folder. `self.totensor`: `torchvision.transforms.ToTensor` transform. `self.transform`: The transforms passed in as `transforms` to the constructor.
#'
#' @param path path to dir
#' @param transforms transformations
#' @return None
#' @export
FolderDataset <- function(path, transforms = NULL) {

  args = list(
    path = path,
    transforms = transforms
  )

  if(is.null(args$transforms))
    args$transforms <- NULL


  do.call(upit()$inference$cyclegan$FolderDataset, args)

}

#' @title Load_dataset
#'
#' @description A helper function for getting a DataLoader for images in the folder `test_path`, with batch size `bs`, and number of workers `num_workers`
#'
#'
#' @param test_path test path (directory)
#' @param bs batch size
#' @param num_workers number of workers
#' @return None
#' @export
load_dataset <- function(test_path, bs = 4, num_workers = 4) {

  upit()$inference$cyclegan$load_dataset(
    test_path = test_path,
    bs = as.integer(bs),
    num_workers = as.integer(num_workers)
  )

}

#' @title Get_preds_cyclegan
#'
#' @description A prediction function that takes the Learner object `learn` with the trained model, the `test_path` folder with the images to perform
#'
#' @details batch inference on, and the output folder `pred_path` where the predictions will be saved, with a batch size `bs`, `num_workers`,
#' and suffix of the prediction images `suffix` (default='png').
#'
#' @param learn learner/model
#' @param test_path testdat path
#' @param pred_path predict data path
#' @param bs batch size
#' @param num_workers number of workers
#' @param suffix suffix
#'
#' @export
get_preds_cyclegan <- function(learn, test_path, pred_path, bs = 4, num_workers = 4, suffix = "tif") {

  upit()$inference$cyclegan$get_preds_cyclegan(
    learn = learn,
    test_path = test_path,
    pred_path = pred_path,
    bs = as.integer(bs),
    num_workers = as.integer(num_workers),
    suffix = suffix
  )

}

#' @title Export_generator
#' @param learn learner/model
#' @param generator_name generator name
#' @param path path (save dir)
#' @param convert_to convert to
#' @return None
#' @export
export_generator <- function(learn, generator_name = "generator", path = '.', convert_to = "B") {

  upit()$inference$cyclegan$export_generator(
    learn = learn,
    generator_name = generator_name,
    path = path,
    convert_to = convert_to
  )

}


