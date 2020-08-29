#' @title AWD_LSTM
#'
#' @description AWD-LSTM inspired by https://arxiv.org/abs/1708.02182
#'
#' @details
#'
#' @param vocab_sz vocab_sz
#' @param emb_sz emb_sz
#' @param n_hid n_hid
#' @param n_layers n_layers
#' @param pad_token pad_token
#' @param hidden_p hidden_p
#' @param input_p input_p
#' @param embed_p embed_p
#' @param weight_p weight_p
#' @param bidir bidir
#'
#' @export
AWD_LSTM <- function(vocab_sz, emb_sz, n_hid, n_layers, pad_token = 1,
                     hidden_p = 0.2, input_p = 0.6, embed_p = 0.1,
                     weight_p = 0.5, bidir = FALSE) {

  if(missing(vocab_sz) & missing(emb_sz) & missing(n_hid) & missing(n_layers)) {
    text$AWD_LSTM
  } else {
    args <- list(
      vocab_sz = vocab_sz,
      emb_sz = emb_sz,
      n_hid = n_hid,
      n_layers = n_layers,
      pad_token = as.integer(pad_token),
      hidden_p = hidden_p,
      input_p = input_p,
      embed_p = embed_p,
      weight_p = weight_p,
      bidir = bidir
    )

    do.call(text$AWD_LSTM, args)
  }

}


#' @title language_model_learner
#'
#' @description Create a `Learner` with a language model from `dls` and `arch`.
#'
#' @details
#'
#' @param dls dls
#' @param arch arch
#' @param config config
#' @param drop_mult drop_mult
#' @param backwards backwards
#' @param pretrained pretrained
#' @param pretrained_fnames pretrained_fnames
#' @param loss_func loss_func
#' @param opt_func opt_func
#' @param lr lr
#' @param splitter splitter
#' @param cbs cbs
#' @param metrics metrics
#' @param path path
#' @param model_dir model_dir
#' @param wd wd
#' @param wd_bn_bias wd_bn_bias
#' @param train_bn train_bn
#' @param moms moms
#'
#' @export
language_model_learner <- function(dls, arch, config = NULL, drop_mult = 1.0,
                                   backwards = FALSE, pretrained = TRUE,
                                   pretrained_fnames = NULL,
                                   opt_func = Adam(), lr = 0.001,
                                   cbs = NULL, metrics = NULL, path = NULL,
                                   model_dir = "models", wd = NULL, wd_bn_bias = FALSE,
                                   train_bn = TRUE, moms = list(0.95, 0.85, 0.95),
                                   ...) {

  args <- list(
    dls = dls,
    arch = arch,
    config = config,
    drop_mult = drop_mult,
    backwards = backwards,
    pretrained = pretrained,
    pretrained_fnames = pretrained_fnames,
    opt_func = opt_func,
    lr = lr,
    cbs = cbs,
    metrics = metrics,
    path = path,
    model_dir = model_dir,
    wd = wd,
    wd_bn_bias = wd_bn_bias,
    train_bn = train_bn,
    moms = moms,
    ...
  )

  do.call(text$language_model_learner, args)

}

#' @title get_text_files
#'
#' @description Get text files in `path` recursively, only in `folders`, if specified.
#'
#' @details
#'
#' @param path path
#' @param recurse recurse
#' @param folders folders
#'
#' @export
get_text_files <- function(path, recurse = TRUE, folders = NULL) {

  if(missing(path)) {
    text$get_text_files
  } else {
    text$get_text_files(
      path = path,
      recurse = recurse,
      folders = folders
    )
  }

}



