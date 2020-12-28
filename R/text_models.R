#' @title AWD_LSTM
#'
#' @description AWD-LSTM inspired by https://arxiv.org/abs/1708.02182
#'
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
#' @return None
#' @export
AWD_LSTM <- function(vocab_sz, emb_sz, n_hid, n_layers, pad_token = 1,
                     hidden_p = 0.2, input_p = 0.6, embed_p = 0.1,
                     weight_p = 0.5, bidir = FALSE) {

  if(missing(vocab_sz) & missing(emb_sz) & missing(n_hid) & missing(n_layers)) {
    text()$AWD_LSTM
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

    do.call(text()$AWD_LSTM, args)
  }

}


#' @title Language_model_learner
#'
#' @description Create a `Learner` with a language model from `dls` and `arch`.
#'
#'
#' @param dls dls
#' @param arch arch
#' @param config config
#' @param drop_mult drop_mult
#' @param backwards backwards
#' @param pretrained pretrained
#' @param pretrained_fnames pretrained_fnames
#' @param opt_func opt_func
#' @param lr lr
#' @param cbs cbs
#' @param metrics metrics
#' @param path path
#' @param model_dir model_dir
#' @param wd wd
#' @param wd_bn_bias wd_bn_bias
#' @param train_bn train_bn
#' @param moms moms
#' @param ... additional arguments
#' @return None
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

  strings = c('config', 'pretrained_fnames', 'cbs', 'metrics', 'path', 'wd')

  for(i in 1:length(strings)) {
    if(is.null(args[[strings[i]]]))
      args[[strings[i]]] <- NULL
  }

  do.call(text()$language_model_learner, args)

}

#' @title Get_text_files
#'
#' @description Get text files in `path` recursively, only in `folders`, if specified.
#'
#'
#' @param path path
#' @param recurse recurse
#' @param folders folders
#' @return None
#' @export
get_text_files <- function(path, recurse = TRUE, folders = NULL) {

  if(missing(path)) {
    text()$get_text_files
  } else {
    text()$get_text_files(
      path = path,
      recurse = recurse,
      folders = folders
    )
  }

}


#' @title LinearDecoder
#'
#' @description To go on top of a RNNCore module and create a Language Model.
#'
#'
#' @param n_out n_out
#' @param n_hid n_hid
#' @param output_p output_p
#' @param tie_encoder tie_encoder
#' @param bias bias
#' @return None
#' @export
LinearDecoder <- function(n_out, n_hid, output_p = 0.1, tie_encoder = NULL, bias = TRUE) {

  text()$LinearDecoder(
    n_out = n_out,
    n_hid = n_hid,
    output_p = output_p,
    tie_encoder = tie_encoder,
    bias = bias
  )

}


#' @title Sequential RNN
#'
#'
#' @param ... parameters to pass
#' @return layer
#' @export
SequentialRNN <- function(...) {
  args = list(...)

  do.call(text()$SequentialRNN, args)
}


#' @title Get_language_model
#'
#' @description Create a language model from `arch` and its `config`.
#'
#'
#' @param arch arch
#' @param vocab_sz vocab_sz
#' @param config config
#' @param drop_mult drop_mult
#' @return model
#' @export
get_language_model <- function(arch, vocab_sz, config = NULL, drop_mult = 1.0) {

  text()$get_language_model(
    arch = arch,
    vocab_sz = vocab_sz,
    config = config,
    drop_mult = drop_mult
  )

}


#' @title SentenceEncoder
#'
#' @description Create an encoder over `module` that can process a full sentence.
#'
#'
#' @param bptt bptt
#' @param module module
#' @param pad_idx pad_idx
#' @param max_len max_len
#' @return None
#' @export
SentenceEncoder <- function(bptt, module, pad_idx = 1, max_len = NULL) {

  args = list(
    bptt = bptt,
    module = module,
    pad_idx = as.integer(pad_idx),
    max_len = max_len
  )

  if(is.null(args$max_len))
    args$max_len <- NULL
  else
    args$max_len <- as.integer(args$max_len)

  do.call(text()$SentenceEncoder, args)
}


#' @title Masked_concat_pool
#'
#' @description Pool `MultiBatchEncoder` outputs into one vector [last_hidden, max_pool, avg_pool]
#'
#'
#' @param output output
#' @param mask mask
#' @param bptt bptt
#' @return None
#' @export
masked_concat_pool <- function(output, mask, bptt) {

  text()$masked_concat_pool(
    output = output,
    mask = mask,
    bptt = bptt
  )

}

#' @title PoolingLinearClassifier
#'
#' @description Create a linear classifier with pooling
#'
#'
#' @param dims dims
#' @param ps ps
#' @param bptt bptt
#' @param y_range y_range
#' @return None
#' @export
PoolingLinearClassifier <- function(dims, ps, bptt, y_range = NULL) {

  args =list(
    dims = dims,
    ps = ps,
    bptt = bptt,
    y_range = y_range
  )

  if(is.null(args$y_range))
    args$y_range <- NULL


  do.call(text()$PoolingLinearClassifier, args)

}

#' @title Get_text_classifier
#'
#' @description Create a text classifier from `arch` and its `config`, maybe `pretrained`
#'
#'
#' @param arch arch
#' @param vocab_sz vocab_sz
#' @param n_class n_class
#' @param seq_len seq_len
#' @param config config
#' @param drop_mult drop_mult
#' @param lin_ftrs lin_ftrs
#' @param ps ps
#' @param pad_idx pad_idx
#' @param max_len max_len
#' @param y_range y_range
#' @return None
#' @export
get_text_classifier <- function(arch, vocab_sz, n_class, seq_len = 72,
                                config = NULL, drop_mult = 1.0,
                                lin_ftrs = NULL, ps = NULL,
                                pad_idx = 1, max_len = 1440,
                                y_range = NULL) {

  args = list(
    arch = arch,
    vocab_sz = vocab_sz,
    n_class = n_class,
    seq_len = as.integer(seq_len),
    config = config,
    drop_mult = drop_mult,
    lin_ftrs = lin_ftrs,
    ps = ps,
    pad_idx = as.integer(pad_idx),
    max_len = as.integer(max_len),
    y_range = y_range
  )

  strings = c('config', 'ps', 'lin_ftrs', 'y_range')

  for(i in 1:length(strings)) {
    if(is.null(args[[strings[i]]]))
      args[[strings[i]]] <- NULL
  }


  do.call(text()$get_text_classifier, args)
}


#' @title Dropout_mask
#'
#' @description Return a dropout mask of the same type as `x`, size `sz`, with probability `p` to cancel an element.
#'
#'
#' @param x x
#' @param sz sz
#' @param p p
#' @return None
#' @export
dropout_mask <- function(x, sz, p) {

  text()$dropout_mask(
    x = x,
    sz = sz,
    p = p
  )

}



#' @title RNNDropout
#'
#' @description Dropout with probability `p` that is consistent on the seq_len dimension.
#'
#'
#' @param p p
#' @return None
#' @export
RNNDropout <- function(p = 0.5) {

  text()$RNNDropout(
    p = p
  )

}


#' @title WeightDropout
#'
#' @description A module that wraps another layer in which some weights will be replaced by 0 during training.
#'
#'
#' @param module module
#' @param weight_p weight_p
#' @param layer_names layer_names
#' @return None
#' @export
WeightDropout <- function(module, weight_p, layer_names = "weight_hh_l0") {

  text()$WeightDropout(
    module = module,
    weight_p = weight_p,
    layer_names = layer_names
  )

}



#' @title EmbeddingDropout
#'
#' @description Apply dropout with probability `embed_p` to an embedding layer `emb`.
#'
#'
#' @param emb emb
#' @param embed_p embed_p
#' @return None
#' @export
EmbeddingDropout <- function(emb, embed_p) {

  text()$EmbeddingDropout(
    emb = emb,
    embed_p = embed_p
  )

}

#' @title Awd_lstm_lm_split
#'
#' @description Split a RNN `model` in groups for differential learning rates.
#'
#'
#' @param model model
#' @return None
#' @export
awd_lstm_lm_split <- function(model) {

  text()$awd_lstm_lm_split(
    model = model
  )

}


#' @title Awd_lstm_clas_split
#'
#' @description Split a RNN `model` in groups for differential learning rates.
#'
#'
#' @param model model
#' @return None
#' @export
awd_lstm_clas_split <- function(model) {

  text()$awd_lstm_clas_split(
    model = model
  )

}


#' @title AWD_QRNN
#'
#' @description Same as an AWD-LSTM, but using QRNNs instead of LSTMs
#'
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
#' @return None
#' @export
AWD_QRNN <- function(vocab_sz, emb_sz, n_hid, n_layers, pad_token = 1,
                     hidden_p = 0.2, input_p = 0.6, embed_p = 0.1,
                     weight_p = 0.5, bidir = FALSE) {

  python_function_result <- text()$AWD_QRNN(
    vocab_sz = as.integer(vocab_sz),
    emb_sz = as.integer(emb_sz),
    n_hid = as.integer(n_hid),
    n_layers = as.integer(n_layers),
    pad_token = as.integer(pad_token),
    hidden_p = hidden_p,
    input_p = input_p,
    embed_p = embed_p,
    weight_p = weight_p,
    bidir = bidir
  )

}




#' @title Forget_mult_CPU
#'
#' @description ForgetMult gate applied to `x` and `f` on the CPU.
#'
#'
#' @param x x
#' @param f f
#' @param first_h first_h
#' @param batch_first batch_first
#' @param backward backward
#' @return None
#' @export
forget_mult_CPU <- function(x, f, first_h = NULL, batch_first = TRUE, backward = FALSE) {

  args= list(
    x = x,
    f = f,
    first_h = first_h,
    batch_first = batch_first,
    backward = backward
  )

  if(is.null(args$first_h))
    args$first_h <- NULL


  do.call(fastai2$text$models$qrnn$forget_mult_CPU, args)

}


#' @title ForgetMultGPU
#'
#' @description Wrapper around the CUDA kernels for the ForgetMult gate.
#'
#' @param ... parameters to pass
#' @return None
#' @export
ForgetMultGPU <- function(...) {

  invisible(fastai2$text$models$qrnn$ForgetMultGPU(...
  ) )

}

#' @title QRNNLayer
#'
#' @description Apply a single layer Quasi-Recurrent Neural Network (QRNN) to an input sequence.
#'
#'
#' @param input_size input_size
#' @param hidden_size hidden_size
#' @param save_prev_x save_prev_x
#' @param zoneout zoneout
#' @param window window
#' @param output_gate output_gate
#' @param batch_first batch_first
#' @param backward backward
#' @return None
#' @export
QRNNLayer <- function(input_size, hidden_size = NULL, save_prev_x = FALSE,
                      zoneout = 0, window = 1, output_gate = TRUE,
                      batch_first = TRUE, backward = FALSE) {

  args = list(
    input_size = input_size,
    hidden_size = hidden_size,
    save_prev_x = save_prev_x,
    zoneout = as.integer(zoneout),
    window = as.integer(window),
    output_gate = output_gate,
    batch_first = batch_first,
    backward = backward
  )

  if(is.null(args$hidden_size))
    args$hidden_size <- NULL

  do.call(fastai2$text$models$qrnn$QRNNLayer, args)
}


#' @title QRNN
#'
#' @description Apply a multiple layer Quasi-Recurrent Neural Network (QRNN) to an input sequence.
#'
#'
#' @param input_size input_size
#' @param hidden_size hidden_size
#' @param n_layers n_layers
#' @param batch_first batch_first
#' @param dropout dropout
#' @param bidirectional bidirectional
#' @param save_prev_x save_prev_x
#' @param zoneout zoneout
#' @param window window
#' @param output_gate output_gate
#' @return None
#' @export
QRNN <- function(input_size, hidden_size, n_layers = 1, batch_first = TRUE,
                 dropout = 0, bidirectional = FALSE, save_prev_x = FALSE,
                 zoneout = 0, window = NULL, output_gate = TRUE) {

  args = list(
    input_size = input_size,
    hidden_size = hidden_size,
    n_layers = as.integer(n_layers),
    batch_first = batch_first,
    dropout = as.integer(dropout),
    bidirectional = bidirectional,
    save_prev_x = save_prev_x,
    zoneout = as.integer(zoneout),
    window = window,
    output_gate = output_gate
  )

  if(is.null(args$window))
    args$window <- NULL

  do.call(fastai2$text$models$qrnn$QRNN, args)

}











