
#' @title Reverse_text
#'
#'
#' @param x text
#' @return string
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

#' @title Make_vocab
#'
#' @description Create a vocab of `max_vocab` size from `Counter` `count` with items present more than `min_freq`
#'
#' @param count count
#' @param min_freq min_freq
#' @param max_vocab max_vocab
#' @param special_toks special_toks
#' @return None
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
#'
#' @param vocab vocab
#' @param min_freq min_freq
#' @param max_vocab max_vocab
#' @param special_toks special_toks
#' @param pad_tok pad_tok
#' @return None
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
#' @return text loader
#' @export
LMDataLoader <- function(dataset, lens = NULL, cache = 2, bs = 64,
                         seq_len = 72, num_workers = 0, shuffle = FALSE,
                         verbose = FALSE, do_setup = TRUE, pin_memory = FALSE,
                         timeout = 0L, batch_size = NULL, drop_last = FALSE,
                         indexed = NULL, n = NULL, device = NULL) {


  if(missing(dataset)) {
    text$LMDataLoader
  } else {
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

}


#' @title LMLearner
#'
#' @description Add functionality to `TextLearner` when dealingwith a language model
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
#' @return text loader
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

#' @title Pad_input
#'
#' @description Function that collect `samples` and adds padding
#'
#'
#' @param samples samples
#' @param pad_idx pad_idx
#' @param pad_fields pad_fields
#' @param pad_first pad_first
#' @param backwards backwards
#' @return None
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


#' @title Pad_input_chunk
#'
#' @description Pad `samples` by adding padding by chunks of size `seq_len`
#'
#'
#' @param samples samples
#' @param pad_idx pad_idx
#' @param pad_first pad_first
#' @param seq_len seq_len
#' @return None
#' @export
pad_input_chunk <- function(samples, pad_idx = 1, pad_first = TRUE, seq_len = 72) {

  args <- list(
    samples = samples,
    pad_idx = as.integer(pad_idx),
    pad_first = pad_first,
    seq_len = as.integer(seq_len)
  )

  do.call(text$pad_input_chunk, args)

}


#' @title SortedDL
#'
#' @description A `DataLoader` that goes throught the item in the order given by `sort_func`
#'
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
#' @return None
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

  if(!is.null(args$batch_size)) {
    args$batch_size = as.integer(args$batch_size)
  }

  do.call(text$SortedDL, args)

}


#' @title TextBlock
#'
#' @description A `TransformBlock` for texts
#'
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
#' @return block object
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

#' @title TextBlock_from_df
#'
#' @description Build a `TextBlock` from a dataframe using `text_cols`
#'
#' @param text_cols text columns
#' @param vocab vocabulary
#' @param is_lm is_lm
#' @param seq_len sequence length
#' @param backwards backwards
#' @param min_freq minimum frequency
#' @param max_vocab max vocabulary
#' @param tok tokenizer
#' @param rules rules
#' @param sep separator
#' @param n_workers number workers
#' @param mark_fields mark_fields
#' @param res_col_name result column name
#' @return None
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


#' @title TextBlock_from_folder
#'
#' @description Build a `TextBlock` from a `path`
#'
#'
#' @param path path
#' @param vocab vocabualry
#' @param is_lm is_lm
#' @param seq_len sequence length
#' @param backwards backwards
#' @param min_freq minimum frequency
#' @param max_vocab max vocabulary
#' @param tok tokenizer
#' @param rules rules
#' @param extensions extensions
#' @param folders folders
#' @param output_dir output_dir
#' @param skip_if_exists skip_if_exists
#' @param output_names output_names
#' @param n_workers number of workers
#' @param encoding encoding
#' @return None
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





#' @title ColReader
#'
#' @description Read `cols` in `row` with potential `pref` and `suff`
#'
#'
#' @param cols columns
#' @param pref pref
#' @param suff suffix
#' @param label_delim label separator
#' @return None
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
#' @param col column
#' @return None
#'
#' @export
ColSplitter <- function(col = "is_valid") {

  text$ColSplitter(
    col = col
  )

}


#' @title TextDataLoaders_from_df
#'
#' @description Create from `df` in `path` with `valid_pct`
#''
#' @param df df
#' @param path path
#' @param valid_pct validation percentage
#' @param seed seed
#' @param text_col text_col
#' @param label_col label_col
#' @param label_delim label_delim
#' @param y_block y_block
#' @param text_vocab text_vocab
#' @param is_lm is_lm
#' @param valid_col valid_col
#' @param tok_tfm tok_tfm
#' @param seq_len seq_len
#' @param backwards backwards
#' @param bs batch size
#' @param val_bs validation batch size, if not specified then val_bs is the same as bs.
#' @param shuffle_train shuffle_train
#' @param device device
#' @return text loader
#' @export
TextDataLoaders_from_df <- function(df, path = ".", valid_pct = 0.2, seed = NULL,
                    text_col = 0, label_col = 1, label_delim = NULL,
                    y_block = NULL, text_vocab = NULL, is_lm = FALSE,
                    valid_col = NULL, tok_tfm = NULL, seq_len = 72,
                    backwards = FALSE, bs = 64, val_bs = NULL,
                    shuffle_train = TRUE, device = NULL) {

  args <- list(
    df = df,
    path = path,
    valid_pct = valid_pct,
    seed = seed,
    text_col = text_col,
    label_col = label_col,
    label_delim = label_delim,
    y_block = y_block,
    text_vocab = text_vocab,
    is_lm = is_lm,
    valid_col = valid_col,
    tok_tfm = tok_tfm,
    seq_len = as.integer(seq_len),
    backwards = backwards,
    bs = as.integer(bs),
    val_bs = val_bs,
    shuffle_train = shuffle_train,
    device = device
  )

  if(is.numeric(text_col))
    args$text_col <- as.integer(args$text_col)
  if(is.numeric(label_col))
    args$label_col <- as.integer(args$label_col)
  if(!is.null(args$val_bs))
    args$val_bs = as.integer(args$val_bs)

  do.call(text$TextDataLoaders$from_df, args)

}


#' @title TextDataLoaders_from_folder
#'
#' @description Create from imagenet style dataset in `path` with `train` and `valid` subfolders (or provide `valid_pct`)
#'
#'
#' @param path path
#' @param train train data
#' @param valid validation data
#' @param valid_pct validation percentage
#' @param seed random seed
#' @param vocab vocabulary
#' @param text_vocab text_vocab
#' @param is_lm is_lm
#' @param tok_tfm tok_tfm
#' @param seq_len seq_len
#' @param backwards backwards
#' @param bs batch size
#' @param val_bs validation batch size
#' @param shuffle_train shuffle train data
#' @param device device
#' @return text loader
#' @export
TextDataLoaders_from_folder <- function(path, train = "train", valid = "valid",
                        valid_pct = NULL, seed = NULL, vocab = NULL,
                        text_vocab = NULL, is_lm = FALSE, tok_tfm = NULL,
                        seq_len = 72, backwards = FALSE, bs = 64,
                        val_bs = NULL, shuffle_train = TRUE, device = NULL) {

  args <- list(
    path = path,
    train = train,
    valid = valid,
    valid_pct = valid_pct,
    seed = seed,
    vocab = vocab,
    text_vocab = text_vocab,
    is_lm = is_lm,
    tok_tfm = tok_tfm,
    seq_len = as.integer(seq_len),
    backwards = backwards,
    bs = as.integer(bs),
    val_bs = val_bs,
    shuffle_train = shuffle_train,
    device = device
  )

  if(!is.null(args$val_bs))
    args$val_bs = as.integer(args$val_bs)

  do.call(text$TextDataLoaders$from_folder, args)

}


#' @title TextDataLoaders_from_csv
#'
#' @description Create from `csv` file in `path/csv_fname`
#'
#' @param path path
#' @param csv_fname csv file name
#' @param header header
#' @param delimiter delimiter
#' @param valid_pct valid_ation percentage
#' @param seed random seed
#' @param text_col text column
#' @param label_col label column
#' @param label_delim label separator
#' @param y_block y_block
#' @param text_vocab text vocabulary
#' @param is_lm is_lm
#' @param valid_col valid column
#' @param tok_tfm tok_tfm
#' @param seq_len seq_len
#' @param backwards backwards
#' @param bs batch size
#' @param val_bs validation batch size
#' @param shuffle_train shuffle train data
#' @param device device
#' @return text loader
#' @export
TextDataLoaders_from_csv <- function(path, csv_fname = "labels.csv",
                     header = "infer", delimiter = NULL, valid_pct = 0.2,
                     seed = NULL, text_col = 0, label_col = 1,
                     label_delim = NULL, y_block = NULL, text_vocab = NULL,
                     is_lm = FALSE, valid_col = NULL, tok_tfm = NULL, seq_len = 72,
                     backwards = FALSE, bs = 64, val_bs = NULL,
                     shuffle_train = TRUE, device = NULL) {

  args <- list(
    path = path,
    csv_fname = csv_fname,
    header = header,
    delimiter = delimiter,
    valid_pct = valid_pct,
    seed = seed,
    text_col = as.integer(text_col),
    label_col = as.integer(label_col),
    label_delim = label_delim,
    y_block = y_block,
    text_vocab = text_vocab,
    is_lm = is_lm,
    valid_col = valid_col,
    tok_tfm = tok_tfm,
    seq_len = seq_len,
    backwards = backwards,
    bs = as.integer(bs),
    val_bs = val_bs,
    shuffle_train = shuffle_train,
    device = device
  )

  if(!is.null(args$val_bs))
    args$val_bs = as.integer(args$val_bs)

  do.call(text$TextDataLoaders$from_csv, args)

}



#' @title RandomSplitter
#'
#' @description Create function that splits `items` between train/val with `valid_pct` randomly.
#'
#'
#' @param valid_pct validation percenatge split
#' @param seed random seed
#' @return None
#' @export
RandomSplitter <- function(valid_pct = 0.2, seed = NULL) {

  text$RandomSplitter(
    valid_pct = valid_pct,
    seed = seed
  )

}





