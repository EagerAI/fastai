
#' @title reverse_text
#'
#'
#' @param x x
#'
#' @export
reverse_text <- function(x) {


  if(missing(x)) {
    text$reverse_text
  } else {
    text$reverse_text(
      x = x
    )
  }

}

#' @title make_vocab
#'
#' @description Create a vocab of `max_vocab` size from `Counter` `count` with items present more than `min_freq`
#'
#' @param count count
#' @param min_freq min_freq
#' @param max_vocab max_vocab
#' @param special_toks special_toks
#'
#' @export
make_vocab <- function(count, min_freq = 3, max_vocab = 60000, special_toks = NULL) {

  args <- list(
    count = count,
    min_freq = as.integer(min_freq),
    max_vocab = as.integer(max_vocab),
    special_toks = special_toks
  )

  do.call(text$make_vocab, args)

}


#' @title Numericalize
#'
#' @description Reversible transform of tokenized texts to numericalized ids
#'
#' @details
#'
#' @param vocab vocab
#' @param min_freq min_freq
#' @param max_vocab max_vocab
#' @param special_toks special_toks
#' @param pad_tok pad_tok
#'
#' @export
Numericalize <- function(vocab = NULL, min_freq = 3, max_vocab = 60000, special_toks = NULL, pad_tok = NULL) {

  args <- list(
    vocab = vocab,
    min_freq = as.integer(min_freq),
    max_vocab = as.integer(max_vocab),
    special_toks = special_toks,
    pad_tok = pad_tok
  )

  do.call(text$Numericalize, args)

}

#' @title LMDataLoader
#'
#' @description A `DataLoader` suitable for language modeling
#'
#' @details
#'
#' @param dataset dataset
#' @param lens lens
#' @param cache cache
#' @param bs bs
#' @param seq_len seq_len
#' @param num_workers num_workers
#' @param shuffle shuffle
#' @param verbose verbose
#' @param do_setup do_setup
#' @param pin_memory pin_memory
#' @param timeout timeout
#' @param batch_size batch_size
#' @param drop_last drop_last
#' @param indexed indexed
#' @param n n
#' @param device device
#'
#' @export
LMDataLoader <- function(dataset, lens = NULL, cache = 2, bs = 64,
                         seq_len = 72, num_workers = 0, shuffle = FALSE,
                         verbose = FALSE, do_setup = TRUE, pin_memory = FALSE,
                         timeout = 0L, batch_size = NULL, drop_last = FALSE,
                         indexed = NULL, n = NULL, device = NULL) {

  args <- list(
    dataset = dataset,
    lens = lens,
    cache = as.integer(cache),
    bs = as.integer(bs),
    seq_len = as.integer(seq_len),
    num_workers = as.integer(num_workers),
    shuffle = shuffle,
    verbose = verbose,
    do_setup = do_setup,
    pin_memory = pin_memory,
    timeout = timeout,
    batch_size = batch_size,
    drop_last = drop_last,
    indexed = indexed,
    n = n,
    device = device
  )

  do.call(text$LMDataLoader, args)

}


#' @title LMLearner
#'
#' @description Add functionality to `TextLearner` when dealingwith a language model
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
LMLearner <- function(dls, model, alpha = 2.0, beta = 1.0,
                      moms = list(0.8, 0.7, 0.8), loss_func = NULL,
                      opt_func = Adam(), lr = 0.001, splitter = trainable_params(),
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

  do.call(text$LMLearner, args)

}

#' @title pad_input
#'
#' @description Function that collect `samples` and adds padding
#'
#'
#' @param samples samples
#' @param pad_idx pad_idx
#' @param pad_fields pad_fields
#' @param pad_first pad_first
#' @param backwards backwards
#'
#' @export
pad_input <- function(samples, pad_idx = 1, pad_fields = 0, pad_first = FALSE, backwards = FALSE) {

  args <- list(
    samples = samples,
    pad_idx = as.integer(pad_idx),
    pad_fields = as.integer(pad_fields),
    pad_first = pad_first,
    backwards = backwards
  )

  do.call(text$pad_input, args)

}


#' @title pad_input_chunk
#'
#' @description Pad `samples` by adding padding by chunks of size `seq_len`
#'
#' @details
#'
#' @param samples samples
#' @param pad_idx pad_idx
#' @param pad_first pad_first
#' @param seq_len seq_len
#'
#' @export
pad_input_chunk <- function(samples, pad_idx = 1L, pad_first = TRUE, seq_len = 72L) {

  args <- list(
    samples = samples,
    pad_idx = pad_idx,
    pad_first = pad_first,
    seq_len = seq_len
  )

  do.call(text$pad_input_chunk, args)

}


#' @title SortedDL
#'
#' @description A `DataLoader` that goes throught the item in the order given by `sort_func`
#'
#' @details
#'
#' @param dataset dataset
#' @param sort_func sort_func
#' @param res res
#' @param bs bs
#' @param shuffle shuffle
#' @param num_workers num_workers
#' @param verbose verbose
#' @param do_setup do_setup
#' @param pin_memory pin_memory
#' @param timeout timeout
#' @param batch_size batch_size
#' @param drop_last drop_last
#' @param indexed indexed
#' @param n n
#' @param device device
#'
#' @export
SortedDL <- function(dataset, sort_func = NULL, res = NULL, bs = 64,
                     shuffle = FALSE, num_workers = NULL, verbose = FALSE,
                     do_setup = TRUE, pin_memory = FALSE, timeout = 0,
                     batch_size = NULL, drop_last = FALSE, indexed = NULL,
                     n = NULL, device = NULL) {

  args <- list(
    dataset = dataset,
    sort_func = sort_func,
    res = res,
    bs = as.integer(bs),
    shuffle = shuffle,
    num_workers = num_workers,
    verbose = verbose,
    do_setup = do_setup,
    pin_memory = pin_memory,
    timeout = as.integer(timeout),
    batch_size = batch_size,
    drop_last = drop_last,
    indexed = indexed,
    n = n,
    device = device
  )

  do.call(text$SortedDL, args)

}


#' @title TextBlock
#'
#' @description A `TransformBlock` for texts
#'
#' @details
#'
#' @param tok_tfm tok_tfm
#' @param vocab vocab
#' @param is_lm is_lm
#' @param seq_len seq_len
#' @param backwards backwards
#' @param min_freq min_freq
#' @param max_vocab max_vocab
#' @param special_toks special_toks
#' @param pad_tok pad_tok
#'
#' @export
TextBlock <- function(tok_tfm, vocab = NULL, is_lm = FALSE, seq_len = 72,
                      backwards = FALSE, min_freq = 3, max_vocab = 60000,
                      special_toks = NULL, pad_tok = NULL) {

  args <- list(
    tok_tfm = tok_tfm,
    vocab = vocab,
    is_lm = is_lm,
    seq_len = as.integer(seq_len),
    backwards = backwards,
    min_freq = as.integer(min_freq),
    max_vocab = as.integer(max_vocab),
    special_toks = special_toks,
    pad_tok = pad_tok
  )

  do.call(text$TextBlock, args)

}

#' @title from_df
#'
#' @description Build a `TextBlock` from a dataframe using `text_cols`
#'
#' @details
#'
#' @param cls cls
#' @param text_cols text_cols
#' @param vocab vocab
#' @param is_lm is_lm
#' @param seq_len seq_len
#' @param backwards backwards
#' @param min_freq min_freq
#' @param max_vocab max_vocab
#' @param tok tok
#' @param rules rules
#' @param sep sep
#' @param n_workers n_workers
#' @param mark_fields mark_fields
#' @param res_col_name res_col_name
#'
#' @export
TextBlock_from_df <- function(text_cols, vocab = NULL, is_lm = FALSE,
                              seq_len = 72, backwards = FALSE, min_freq = 3,
                              max_vocab = 60000, tok = NULL, rules = NULL,
                              sep = " ", n_workers = 6, mark_fields = NULL,
                              res_col_name = "text") {

  args <- list(
    text_cols = text_cols,
    vocab = vocab,
    is_lm = is_lm,
    seq_len = as.integer(seq_len),
    backwards = backwards,
    min_freq = as.integer(min_freq),
    max_vocab = as.integer(max_vocab),
    tok = tok,
    rules = rules,
    sep = sep,
    n_workers = as.integer(n_workers),
    mark_fields = mark_fields,
    res_col_name = res_col_name
  )

  do.call(text$TextBlock$from_df, args)

}


#' @title from_folder
#'
#' @description Build a `TextBlock` from a `path`
#'
#'
#' @param path path
#' @param vocab vocab
#' @param is_lm is_lm
#' @param seq_len seq_len
#' @param backwards backwards
#' @param min_freq min_freq
#' @param max_vocab max_vocab
#' @param tok tok
#' @param rules rules
#' @param extensions extensions
#' @param folders folders
#' @param output_dir output_dir
#' @param skip_if_exists skip_if_exists
#' @param output_names output_names
#' @param n_workers n_workers
#' @param encoding encoding
#'
#' @export
TextBlock_from_folder <- function(path, vocab = NULL, is_lm = FALSE, seq_len = 72,
                                  backwards = FALSE, min_freq = 3, max_vocab = 60000,
                                  tok = NULL, rules = NULL, extensions = NULL, folders = NULL,
                                  output_dir = NULL, skip_if_exists = TRUE, output_names = NULL,
                                  n_workers = 6, encoding = "utf8") {

  args <- list(
    path = path,
    vocab = vocab,
    is_lm = is_lm,
    seq_len = as.integer(seq_len),
    backwards = backwards,
    min_freq = as.integer(min_freq),
    max_vocab = as.integer(max_vocab),
    tok = tok,
    rules = rules,
    extensions = extensions,
    folders = folders,
    output_dir = output_dir,
    skip_if_exists = skip_if_exists,
    output_names = output_names,
    n_workers = as.integer(n_workers),
    encoding = encoding
  )

  do.call(text$TextBlock$from_folder, args)

}




#' @title CategoryBlock
#'
#' @description `TransformBlock` for single-label categorical targets
#'
#'
#' @param vocab vocab
#' @param sort sort
#' @param add_na add_na
#'
#' @export
CategoryBlock <- function(vocab = NULL, sort = TRUE, add_na = FALSE) {

  if(is.null(vocab)) {
    text$CategoryBlock
  } else {
    text$CategoryBlock(
      vocab = vocab,
      sort = sort,
      add_na = add_na
    )
  }

}


#' @title ColReader
#'
#' @description Read `cols` in `row` with potential `pref` and `suff`
#'
#' @details
#'
#' @param cols cols
#' @param pref pref
#' @param suff suff
#' @param label_delim label_delim
#'
#' @export
ColReader <- function(cols, pref = "", suff = "", label_delim = NULL) {

  if(missing(cols)) {
    text$ColReader
  } else {
    text$ColReader(
      cols = cols,
      pref = pref,
      suff = suff,
      label_delim = label_delim
    )
  }

}



#' @title ColSplitter
#'
#' @description Split `items` (supposed to be a dataframe) by value in `col`
#'
#'
#' @param col col
#'
#' @export
ColSplitter <- function(col = "is_valid") {

  text$ColSplitter(
    col = col
  )

}






