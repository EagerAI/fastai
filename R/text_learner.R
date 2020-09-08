
#' @title match_embeds
#'
#' @description Convert the embedding in `old_wgts` to go from `old_vocab` to `new_vocab`.
#'
#'
#' @param old_wgts old_wgts
#' @param old_vocab old_vocab
#' @param new_vocab new_vocab
#'
#' @export
match_embeds <- function(old_wgts, old_vocab, new_vocab) {

  text$match_embeds(
    old_wgts = old_wgts,
    old_vocab = old_vocab,
    new_vocab = new_vocab
  )

}

#' @title load_ignore_keys
#'
#' @description Load `wgts` in `model` ignoring the names of the keys, just taking parameters in order
#'
#'
#' @param model model
#' @param wgts wgts
#'
#' @export
load_ignore_keys <- function(model, wgts) {

  text$load_ignore_keys(
    model = model,
    wgts = wgts
  )

}


#' @title clean_raw_keys
#'
#'
#' @param wgts wgts
#'
#' @export
clean_raw_keys <- function(wgts) {

  text$clean_raw_keys(
    wgts = wgts
  )

}


#' @title load_model_text
#'
#' @description Load `model` from `file` along with `opt` (if available, and if `with_opt`)
#'
#'
#' @param file file
#' @param model model
#' @param opt opt
#' @param with_opt with_opt
#' @param device device
#' @param strict strict
#'
#' @export
load_model_text <- function(file, model, opt, with_opt = NULL, device = NULL, strict = TRUE) {

  text$load_model_text(
    file = file,
    model = model,
    opt = opt,
    with_opt = with_opt,
    device = device,
    strict = strict
  )

}


#' @title TextLearner
#'
#' @description Basic class for a `Learner` in NLP.
#'
#' @details
#'
#' @param dls dls
#' @param model model
#' @param alpha alpha
#' @param beta beta
#' @param moms moms
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
#'
#' @export
TextLearner <- function(dls, model, alpha = 2.0, beta = 1.0,
                        moms = list(0.8, 0.7, 0.8), loss_func = NULL,
                        opt_func = Adam, lr = 0.001, splitter = trainable_params(),
                        cbs = NULL, metrics = NULL, path = NULL, model_dir = "models",
                        wd = NULL, wd_bn_bias = FALSE, train_bn = TRUE) {

  args <- list(
    dls = dls,
    model = model,
    alpha = alpha,
    beta = beta,
    moms = moms,
    loss_func = loss_func,
    opt_func = opt_func,
    lr = lr,
    splitter = splitter,
    cbs = cbs,
    metrics = metrics,
    path = path,
    model_dir = model_dir,
    wd = wd,
    wd_bn_bias = wd_bn_bias,
    train_bn = train_bn
  )

  do.call(text$TextLearner, args)

}


#' @title load_pretrained
#'
#' @description Load a pretrained model and adapt it to the data vocabulary.
#'
#'
#' @param wgts_fname wgts_fname
#' @param vocab_fname vocab_fname
#' @param model model
#'
#' @export
TextLearner_load_pretrained <- function(wgts_fname, vocab_fname, model = NULL) {

  text$TextLearner$load_pretrained(
    wgts_fname = wgts_fname,
    vocab_fname = vocab_fname,
    model = model
  )

}


#' @title save_encoder
#'
#' @description Save the encoder to `file` in the model directory
#'
#'
#' @param file file
#'
#' @export
TextLearner_save_encoder <- function(file) {

  text$TextLearner$save_encoder(
    file = file
  )

}

#' @title load_encoder
#'
#' @description Load the encoder `file` from the model directory, optionally ensuring it's on `device`
#'
#'
#' @param file file
#' @param device device
#'
#' @export
TextLearner_load_encoder <- function(file, device = NULL) {

  text$TextLearner$load_encoder(
    file = file,
    device = device
  )

}


#' @title decode_spec_tokens
#'
#' @description Decode the special tokens in `tokens`
#'
#' @param tokens tokens
#'
#' @export
decode_spec_tokens <- function(tokens) {

  text$decode_spec_tokens(
    tokens = tokens
  )

}


#' @title LMLearner
#'
#' @description Add functionality to `TextLearner` when dealing with a language model
#'
#'
#' @param dls dls
#' @param model model
#' @param alpha alpha
#' @param beta beta
#' @param moms moms
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
#'
#' @export
LMLearner <- function(dls, model, alpha = 2.0, beta = 1.0, moms = list(0.8, 0.7, 0.8),
                      loss_func = NULL, opt_func = Adam(), lr = 0.001,
                      splitter = trainable_params(), cbs = NULL, metrics = NULL,
                      path = NULL, model_dir = "models", wd = NULL,
                      wd_bn_bias = FALSE, train_bn = TRUE) {

  text$LMLearner(
    dls = dls,
    model = model,
    alpha = alpha,
    beta = beta,
    moms = moms,
    loss_func = loss_func,
    opt_func = opt_func,
    lr = lr,
    splitter = splitter,
    cbs = cbs,
    metrics = metrics,
    path = path,
    model_dir = model_dir,
    wd = wd,
    wd_bn_bias = wd_bn_bias,
    train_bn = train_bn
  )

}

#' @title predict
#'
#' @description Return `text` and the `n_words` that come after
#'
#' @param text text
#' @param n_words n_words
#' @param no_unk no_unk
#' @param temperature temperature
#' @param min_p min_p
#' @param no_bar no_bar
#' @param decoder decoder
#' @param only_last_word only_last_word
#'
#' @export
LMLearner_predict <- function(text, n_words = 1, no_unk = TRUE,
                              temperature = 1.0, min_p = NULL, no_bar = FALSE,
                              decoder = decode_spec_tokens(), only_last_word = FALSE) {

 text$LMLearner$predict(
    text = text,
    n_words = as.integer(n_words),
    no_unk = no_unk,
    temperature = temperature,
    min_p = min_p,
    no_bar = no_bar,
    decoder = decoder,
    only_last_word = only_last_word
  )

}




#' @title text_classifier_learner
#'
#' @description Create a `Learner` with a text classifier from `dls` and `arch`.
#'
#'
#' @param dls dls
#' @param arch arch
#' @param seq_len seq_len
#' @param config config
#' @param backwards backwards
#' @param pretrained pretrained
#' @param drop_mult drop_mult
#' @param n_out n_out
#' @param lin_ftrs lin_ftrs
#' @param ps ps
#' @param max_len max_len
#' @param y_range y_range
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
text_classifier_learner <- function(dls, arch, seq_len = 72,
                                    config = NULL, backwards = FALSE,
                                    pretrained = TRUE, drop_mult = 0.5,
                                    n_out = NULL, lin_ftrs = NULL, ps = NULL,
                                    max_len = 1440, y_range = NULL,
                                    loss_func = NULL, opt_func = Adam, lr = 0.001,
                                    splitter = trainable_params, cbs = NULL,
                                    metrics = NULL, path = NULL, model_dir = "models",
                                    wd = NULL, wd_bn_bias = FALSE, train_bn = TRUE,
                                    moms = list(0.95, 0.85, 0.95)) {

  text$text_classifier_learner(
    dls = dls,
    arch = arch,
    seq_len = as.integer(seq_len),
    config = config,
    backwards = backwards,
    pretrained = pretrained,
    drop_mult = drop_mult,
    n_out = n_out,
    lin_ftrs = lin_ftrs,
    ps = ps,
    max_len = as.integer(max_len),
    y_range = y_range,
    loss_func = loss_func,
    opt_func = opt_func,
    lr = lr,
    splitter = splitter,
    cbs = cbs,
    metrics = metrics,
    path = path,
    model_dir = model_dir,
    wd = wd,
    wd_bn_bias = wd_bn_bias,
    train_bn = train_bn,
    moms = moms
  )

}















