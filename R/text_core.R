#' @title spec_add_spaces
#'
#' @description Add spaces around / and #
#'
#' @param t t
#'
#' @export
spec_add_spaces <- function(t) {

  if(missing(t)) {
    invisible(text$spec_add_spaces)
  } else {
    text$spec_add_spaces(
      t = t
    )
  }

}



#' @title Rm_useless_spaces
#'
#' @description Remove multiple spaces
#'
#'
#' @param t t
#'
#' @export
rm_useless_spaces <- function(t) {


  if(missing(t)) {
    invisible(text$rm_useless_spaces)
  } else {
    text$rm_useless_spaces(
      t = t
    )
  }


}

#' @title replace_rep
#'
#' @description Replace repetitions at the character level: cccc -- TK_REP 4 c
#'
#' @details
#'
#' @param t t
#'
#' @export
replace_rep <- function(t) {


  if(missing(t)) {
    invisible(text$replace_rep)
  } else {
    text$replace_rep(
      t = t
    )
  }

}



#' @title replace_wrep
#'
#' @description Replace word repetitions: word word word word -- TK_WREP 4 word
#'
#' @details
#'
#' @param t t
#'
#' @export
replace_wrep <- function(t) {

  if(missing(t)) {
    invisible( text$replace_wrep)
  } else {
    text$replace_wrep(
      t = t
    )
  }
}


#' @title fix_html
#'
#' @description Various messy things we've seen in documents
#'
#' @details
#'
#' @param x x
#'
#' @export
fix_html <- function(x) {



  if(missing(t)) {
    invisible(text$fix_html)
  } else {
    text$fix_html(
      x = x
    )
  }

}

#' @title replace_all_caps
#'
#' @description Replace tokens in ALL CAPS by their lower version and add `TK_UP` before.
#'
#' @details
#'
#' @param t t
#'
#' @export
replace_all_caps <- function(t) {


  if(missing(t)) {
    invisible(text$replace_all_caps)
  } else {
    text$replace_all_caps(
      t = t
    )
  }
}


#' @title replace_maj
#'
#' @description Replace tokens in ALL CAPS by their lower version and add `TK_UP` before.
#'
#'
#' @param t t
#'
#' @export
replace_maj <- function(t) {
  if(missing(t)) {
    invisible(text$replace_maj)
  } else {
    text$replace_maj(
      t = t
    )
  }
}


#' @title BaseTokenizer
#'
#' @description Basic tokenizer that just splits on spaces
#'
#'
#' @param split_char split_char
#'
#' @export
BaseTokenizer <- function(split_char = " ") {

  text$BaseTokenizer(
    split_char = split_char
  )

}


#' @title SpacyTokenizer
#'
#' @description Spacy tokenizer for `lang`
#'
#'
#' @param lang lang
#' @param special_toks special_toks
#' @param buf_sz buf_sz
#'
#' @export
SpacyTokenizer <- function(lang = "en", special_toks = NULL, buf_sz = 5000) {

  text$SpacyTokenizer(
    lang = lang,
    special_toks = special_toks,
    buf_sz = as.integer(buf_sz)
  )

}


#' @title TokenizeWithRules
#'
#' @description A wrapper around `tok` which applies `rules`, then tokenizes, then applies `post_rules`
#'
#'
#' @param tok tok
#' @param rules rules
#' @param post_rules post_rules
#'
#' @export
TokenizeWithRules <- function(tok, rules = NULL, post_rules = NULL) {

  text$TokenizeWithRules(
    tok = tok,
    rules = rules,
    post_rules = post_rules
  )

}

#' @title tokenize1
#'
#' @description Call `TokenizeWithRules` with a single text
#'
#'
#' @param text text
#' @param tok tok
#' @param rules rules
#' @param post_rules post_rules
#'
#' @export
tokenize1 <- function(text, tok, rules = NULL, post_rules = NULL) {

  text$tokenize1(
    text = text,
    tok = tok,
    rules = rules,
    post_rules = post_rules
  )

}

#' @title parallel_tokenize
#'
#' @description Calls optional `setup` on `tok` before launching `TokenizeWithRules` using `parallel_gen
#'
#'
#' @param items items
#' @param tok tok
#' @param rules rules
#' @param n_workers n_workers
#'
#' @export
parallel_tokenize <- function(items, tok = NULL, rules = NULL, n_workers = 6) {

  text$parallel_tokenize(
    items = items,
    tok = tok,
    rules = rules,
    n_workers = as.integer(n_workers)
  )

}



#' @title tokenize_csv
#'
#' @description Tokenize texts in the `text_cols` of the csv `fname` in parallel using `n_workers`
#'
#'
#' @param fname fname
#' @param text_cols text_cols
#' @param outname outname
#' @param n_workers n_workers
#' @param rules rules
#' @param mark_fields mark_fields
#' @param tok tok
#' @param header header
#' @param chunksize chunksize
#'
#' @export
tokenize_csv <- function(fname, text_cols, outname = NULL, n_workers = 4,
                         rules = NULL, mark_fields = NULL, tok = NULL,
                         header = "infer", chunksize = 50000) {

  args <- list(
    fname = fname,
    text_cols = text_cols,
    outname = outname,
    n_workers = as.integer(n_workers),
    rules = rules,
    mark_fields = mark_fields,
    tok = tok,
    header = header,
    chunksize = as.integer(chunksize)
  )

  do.call(text$tokenize_csv, args)

}

#' @title tokenize_df
#'
#' @description Tokenize texts in `df[text_cols]` in parallel using `n_workers`
#'
#' @details
#'
#' @param df df
#' @param text_cols text_cols
#' @param n_workers n_workers
#' @param rules rules
#' @param mark_fields mark_fields
#' @param tok tok
#' @param res_col_name res_col_name
#'
#' @export
tokenize_df <- function(df, text_cols, n_workers = 6, rules = NULL,
                        mark_fields = NULL, tok = NULL,
                        res_col_name = "text") {

  args <-list(
    df = df,
    text_cols = text_cols,
    n_workers = as.integer(n_workers),
    rules = rules,
    mark_fields = mark_fields,
    tok = tok,
    res_col_name = res_col_name
  )

  do.call( text$tokenize_df,args)

}


#' @title tokenize_files
#'
#' @description Tokenize text `files` in parallel using `n_workers`
#'
#' @param files files
#' @param path path
#' @param output_dir output_dir
#' @param output_names output_names
#' @param n_workers n_workers
#' @param rules rules
#' @param tok tok
#' @param encoding encoding
#' @param skip_if_exists skip_if_exists
#'
#' @export
tokenize_files <- function(files, path, output_dir, output_names = NULL,
                           n_workers = 6, rules = NULL, tok = NULL,
                           encoding = "utf8", skip_if_exists = FALSE) {

 args <- list(
    files = files,
    path = path,
    output_dir = output_dir,
    output_names = output_names,
    n_workers = as.integer(n_workers),
    rules = rules,
    tok = tok,
    encoding = encoding,
    skip_if_exists = skip_if_exists
  )

 do.call(text$tokenize_files, args)

}

#' @title tokenize_folder
#'
#' @description Tokenize text files in `path` in parallel using `n_workers`
#'
#'
#' @param path path
#' @param extensions extensions
#' @param folders folders
#' @param output_dir output_dir
#' @param skip_if_exists skip_if_exists
#' @param output_names output_names
#' @param n_workers n_workers
#' @param rules rules
#' @param tok tok
#' @param encoding encoding
#'
#' @export
tokenize_folder <- function(path, extensions = NULL, folders = NULL,
                            output_dir = NULL, skip_if_exists = TRUE,
                            output_names = NULL, n_workers = 6,
                            rules = NULL, tok = NULL, encoding = "utf8") {

  python_function_result <- text$tokenize_folder(
    path = path,
    extensions = extensions,
    folders = folders,
    output_dir = output_dir,
    skip_if_exists = skip_if_exists,
    output_names = output_names,
    n_workers = as.integer(n_workers),
    rules = rules,
    tok = tok,
    encoding = encoding
  )

}

#' @title tokenize_texts
#'
#' @description Tokenize `texts` in parallel using `n_workers`
#'
#'
#' @param texts texts
#' @param n_workers n_workers
#' @param rules rules
#' @param tok tok
#'
#' @export
tokenize_texts <- function(texts, n_workers = 6, rules = NULL, tok = NULL) {

  args <- list(
    texts = texts,
    n_workers = as.integer(n_workers),
    rules = rules,
    tok = tok
  )

  do.call(text$tokenize_texts, args)

}

#' @title load_tokenized_csv
#'
#' @description Utility function to quickly load a tokenized csv and the corresponding counter
#'
#'
#' @param fname fname
#'
#' @export
load_tokenized_csv <- function(fname) {

  text$load_tokenized_csv(
    fname = fname
  )

}


#' @title Tokenizer
#'
#' @description Provides a consistent `Transform` interface to tokenizers operating on `DataFrame`s and folders
#'
#'
#' @param tok tok
#' @param rules rules
#' @param counter counter
#' @param lengths lengths
#' @param mode mode
#' @param sep sep
#'
#' @export
Tokenizer <- function(tok, rules = NULL, counter = NULL, lengths = NULL, mode = NULL, sep = " ") {

  if(missing(tok)) {
    text$Tokenizer
  } else {
    args <- list(
      tok = tok,
      rules = rules,
      counter = counter,
      lengths = lengths,
      mode = mode,
      sep = sep
    )
    do.call(text$Tokenizer, args)
  }

}

#' @title SentencePieceTokenizer
#'
#' @description SentencePiece tokenizer for `lang`
#'
#' @details
#'
#' @param lang lang
#' @param special_toks special_toks
#' @param sp_model sp_model
#' @param vocab_sz vocab_sz
#' @param max_vocab_sz max_vocab_sz
#' @param model_type model_type
#' @param char_coverage char_coverage
#' @param cache_dir cache_dir
#'
#' @export
SentencePieceTokenizer <- function(lang = "en", special_toks = NULL,
                                   sp_model = NULL, vocab_sz = NULL,
                                   max_vocab_sz = 30000L, model_type = "unigram",
                                   char_coverage = NULL, cache_dir = "tmp") {

  args <- list(
    lang = lang,
    special_toks = special_toks,
    sp_model = sp_model,
    vocab_sz = vocab_sz,
    max_vocab_sz = max_vocab_sz,
    model_type = model_type,
    char_coverage = char_coverage,
    cache_dir = cache_dir
  )

  do.call(text$SentencePieceTokenizer, args)

}











