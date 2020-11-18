
#' @title HF_BaseInput
#'
#' @description A HF_BaseInput object is returned from the decodes method
#' of HF_BatchTransform as a mean to customize `@typedispatched` functions
#' like DataLoaders.show_batch and Learner.show_results. It represents the
#' "input_ids" of a huggingface sequence as a tensor with a show method that
#' requires a huggingface tokenizer for proper display.
#'
#' @param ... parameters to pass
#' @return None
#' @export
HF_BaseInput <- function(...) {

  args = list(...)

  if(length(args)>0) {
    blurr()$data$all$HF_BaseInput
  } else {
    do.call(blurr()$data$all$HF_BaseInput, args)
  }

}



#' @title HF_TextBlock
#'
#' @description A basic wrapper that links defaults transforms for the data block API
#'
#'
#' @param hf_arch achitecture
#' @param hf_tokenizer tokenizer
#' @param max_length maximum length
#' @param padding padding or not
#' @param truncation truncation or not
#' @param is_split_into_words to split into words or not
#' @param n_tok_inps number of tok inp
#' @param ... additional arguments
#' @return None
#' @export
HF_TextBlock <- function(hf_arch=NULL, hf_tokenizer=NULL,
                         max_length=512, padding=TRUE, truncation=TRUE, is_split_into_words=FALSE,
                         n_tok_inps=1, ...) {

  args <- list(
    hf_arch=hf_arch,
    hf_tokenizer=hf_tokenizer,
    max_length=as.integer(max_length),
    padding=padding,
    truncation=truncation,
    is_split_into_words=is_split_into_words,
    n_tok_inps=n_tok_inps,
    ...
  )

  if(!is.null(args$max_length)) {
    args$max_length = as.integer(args$max_length)
  }

  if(!is.null(args$n_tok_inps)) {
    args$n_tok_inps = as.integer(args$n_tok_inps)
  }

  do.call(blurr()$data$all$HF_TextBlock, args)

}





#' @title Get_hf_objects
#'
#' @description Returns the architecture (str), config (obj), tokenizer (obj), and model (obj) given at minimum a
#'
#' @details `pre-trained model name or path`. Specify a `task` to ensure the right "AutoModelFor<task>" is used to
#' create the model. Optionally, you can pass a config (obj), tokenizer (class), and/or model (class) (along with any
#' related kwargs for each) to get as specific as you want w/r/t what huggingface objects are returned.
#'
#' @param ... parameters to pass
#' @return None
#' @export
get_hf_objects <- function(...) {

  args <- list(
    ...
  )

  do.call(blurr()$data$all$BLURR_MODEL_HELPER$get_hf_objects, args)

}



#' @title HF_BaseModelWrapper
#'
#' @description Same as `nn.Module`, but no need for subclasses to call `super().__init__`
#'
#'
#' @param hf_model model
#' @param output_hidden_states output hidden states
#' @param output_attentions output attentions
#' @param ... additional arguments to pass
#' @return None
#' @export
HF_BaseModelWrapper <- function(hf_model, output_hidden_states = FALSE,
                                output_attentions = FALSE, ...) {

  blurr()$modeling$all$HF_BaseModelWrapper(
    hf_model = hf_model,
    output_hidden_states = output_hidden_states,
    output_attentions = output_attentions,
    ...
  )

}


#' HF_BaseModelCallback
#'
#'
#' @param ... parameters to pass
#' @return None
#' @export
HF_BaseModelCallback = function(...) {
  args = list(...)

  if(length(args)>0) {
    do.call(blurr()$modeling$all$HF_BaseModelCallback, args)
  } else {
    blurr()$modeling$all$HF_BaseModelCallback
  }
}


#' @title Hf_splitter
#'
#' @description Splits the huggingface model based on various model architecture conventions
#'
#'
#' @param m parameters
#' @return None
#' @export
hf_splitter <- function(m) {

  if(missing(m)) {
    blurr()$modeling$all$hf_splitter
  } else {
    blurr()$modeling$all$hf_splitter(
      m = m
    )
  }

}


#' @title BLURR_MODEL_HELPER
#'
#'
#'
#'
#'
#' @return None
#' @export
helper = function() {
  blurr()$data$all$BLURR_MODEL_HELPER
}


#' Auto configuration
#'
#'
#' @return None
#' @export
AutoConfig = function() {
  if(reticulate::py_module_available('transformers')) {
    trf = reticulate::import('transformers')
    trf$AutoConfig
  }
}





#' @title Load_dataset
#'
#' @description Load a dataset
#'
#' @details This method does the following under the hood:
#' 1. Download and import in the library the dataset loading script from ``path`` if it's not
#' already cached inside the library. Processing scripts are small python scripts that define
#' the citation, info and format of the dataset, contain the URL to the original data files and
#' the code to load examples from the original data files. You can find some of the scripts
#' here: https://github.com/huggingface/datasets/datasets and easily upload yours to share
#' them using the CLI ``datasets-cli``.
#' 2. Run the dataset loading script which will: * Download the dataset file from the original
#' URL (see the script) if it's not already downloaded and cached. * Process and cache
#' the dataset in typed Arrow tables for caching. Arrow table are arbitrarily long, typed
#' tables which can store nested objects and be mapped to numpy/pandas/python standard types.
#' They can be directly access from drive, loaded in RAM or even streamed over the web.
#' 3. Return a dataset build from the requested splits in ``split`` (default: all).
#'
#' @param path path
#' @param name name
#' @param data_dir dataset dir
#' @param data_files dataset files
#' @param split split
#' @param cache_dir cache directory
#' @param features features
#' @param download_config download configuration
#' @param download_mode download mode
#' @param ignore_verifications ignore verifications or not
#' @param save_infos save information or not
#' @param script_version script version
#' @param ... additional arguments
#'
#' @return data frame
#'
#' @export
HF_load_dataset <- function(path, name = NULL, data_dir = NULL, data_files = NULL,
                         split = NULL, cache_dir = NULL, features = NULL,
                         download_config = NULL, download_mode = NULL,
                         ignore_verifications = FALSE, save_infos = FALSE, script_version = NULL, ...) {

  if(reticulate::py_module_available('datasets')) {
    datasets_hug = reticulate::import('datasets')
    #raw_data = datasets_hug$load_dataset('civil_comments', split='train[:1%]')
    args <- list(
      path = path,
      name = name,
      data_dir = data_dir,
      data_files = data_files,
      split = split,
      cache_dir = cache_dir,
      features = features,
      download_config = download_config,
      download_mode = download_mode,
      ignore_verifications = ignore_verifications,
      save_infos = save_infos,
      script_version = script_version,
      ...
    )

    raw_data = do.call(datasets_hug$load_dataset, args)
  }


  invisible(raw_data$data$to_pandas())

}



#' @title Pre_process_squad
#'
#'
#' @param row row in dataframe
#' @param hf_arch architecture
#' @param hf_tokenizer tokenizer
#' @return None
#' @export
pre_process_squad <- function(row, hf_arch, hf_tokenizer) {

  if(missing(row)) {
    blurr()$data$all$pre_process_squad
  } else {
    blurr()$data$all$pre_process_squad(
      row = row,
      hf_arch = hf_arch,
      hf_tokenizer = hf_tokenizer
    )
  }

}

#' @title HF_QABatchTransform
#'
#' @description Handles everything you need to assemble a mini-batch of inputs and targets,
#' as well as decode the dictionary produced
#'
#' @details as a byproduct of the tokenization process in the `encodes` method.
#'
#' @param hf_arch architecture
#' @param hf_tokenizer tokenizer
#' @param max_length maximum length
#' @param padding padding
#' @param truncation truncation
#' @param is_split_into_words to split into words or not
#' @param n_tok_inps number of tok inputs
#' @param hf_input_return_type input return type
#' @param ... additional arguments
#' @return None
#' @export
HF_QABatchTransform <- function(hf_arch, hf_tokenizer, max_length = NULL,
                                padding = TRUE, truncation = TRUE, is_split_into_words = FALSE,
                                n_tok_inps = 1, hf_input_return_type = HF_QuestionAnswerInput(), ...) {

  args <- list(
    hf_arch = hf_arch,
    hf_tokenizer = hf_tokenizer,
    max_length = max_length,
    padding = padding,
    truncation = truncation,
    is_split_into_words = is_split_into_words,
    n_tok_inps = as.integer(n_tok_inps),
    hf_input_return_type = hf_input_return_type,
    ...
  )

  if(!is.null(args[['max_length']])) {
    args[['max_length']] <- as.integer(args[['max_length']])
  }

  do.call(blurr()$data$all$HF_QABatchTransform, args)

}

#' @title HF_QuestionAnswerInput
#'
#'
#' @param ... parameters to apss
#' @return None
#' @export
HF_QuestionAnswerInput <- function(...) {

  args = list(...)

  if(length(args)>0) {
    do.call(blurr()$data$all$HF_QuestionAnswerInput, args)
  } else {
    blurr()$data$all$HF_QuestionAnswerInput
  }

}


#' @title MultiTargetLoss
#'
#' @description Provides the ability to apply different loss functions to multi-modal targets/predictions
#'
#'
#' @param ... additional arguments
#' @return None
#' @export
MultiTargetLoss <- function(...) {

  args <- list(
    ...
  )

  if(!is.null(args[['weights']]) & is.list(args[['weights']])) {
    args[['weights']] <- as.list(as.integer(unlist(args[['weights']])))
  }

  if(!is.null(args[['weights']]) & is.vector(args[['weights']]))
    args[['weights']] <- as.list(as.integer(args[['weights']]))

  do.call(blurr()$modeling$all$MultiTargetLoss, args)

}

#' HF_QstAndAnsModelCallback
#' @param ... parameters to pass
#'
#' @return None
#' @export
HF_QstAndAnsModelCallback <- function(...) {
  args = list(...)
  if(length(args) > 0) {
    do.call(blurr()$modeling$all$HF_QstAndAnsModelCallback, args)
  } else {
    blurr()$modeling$all$HF_QstAndAnsModelCallback
  }
}



#' @title HF_QABeforeBatchTransform
#'
#' @description Handles everything you need to assemble a mini-batch of inputs and targets,
#' as well as decode the dictionary produced
#'
#' @details as a byproduct of the tokenization process in the `encodes` method.
#'
#' @param hf_arch architecture
#' @param hf_tokenizer tokenizer
#' @param max_length maximum length
#' @param padding padding or not
#' @param truncation truncation or not
#' @param is_split_into_words into split into words or not
#' @param n_tok_inps number of tok inputs
#' @param ... additional arguments
#' @return None
#' @export
HF_QABeforeBatchTransform <- function(hf_arch, hf_tokenizer, max_length = NULL,
                                      padding = TRUE, truncation = TRUE, is_split_into_words = FALSE,
                                      n_tok_inps = 1, ...) {

  args <- list(
    hf_arch = hf_arch,
    hf_tokenizer = hf_tokenizer,
    max_length = max_length,
    padding = padding,
    truncation = truncation,
    is_split_into_words = is_split_into_words,
    n_tok_inps = as.integer(n_tok_inps),
    ...
  )

  if(!is.null(args[['max_length']]))
    args[['max_length']] <- as.integer(args[['max_length']])


  do.call(blurr()$data$all$HF_QABeforeBatchTransform, args)

}


#' @title Py_apply
#'
#' @description Pandas apply
#' @param df dataframe
#' @param ... additional arguments
#' @return dataframe
#' @export
py_apply = function(df, ...) {
  py_df = reticulate::r_to_py(df)
  args = list(...)
  py_df = py_df$apply(unlist(args),axis = 1L)
  invisible(reticulate::py_to_r(py_df))
}



#' @title HF_ARCHITECTURES
#'
#' @description An enumeration.
#'
#' @return None
#'
#' @export
HF_ARCHITECTURES <- function() {

  blurr()$data$all$HF_ARCHITECTURES

}




#' @title HF_BeforeBatchTransform
#'
#' @description Handles everything you need to assemble a mini-batch of inputs and targets,
#' as well as decode the dictionary produced
#'
#' @details as a byproduct of the tokenization process in the `encodes` method.
#'
#' @param hf_arch architecture
#' @param hf_tokenizer tokenizer
#' @param max_length maximum length
#' @param padding padding or not
#' @param truncation truncation or not
#' @param is_split_into_words to split into words
#' @param n_tok_inps number tok inputs
#' @param ... additional arguments
#' @return None
#' @export
HF_BeforeBatchTransform <- function(hf_arch, hf_tokenizer, max_length = NULL,
                                    padding = TRUE, truncation = TRUE,
                                    is_split_into_words = FALSE, n_tok_inps = 1, ...) {

  args <- list(
    hf_arch = hf_arch,
    hf_tokenizer = hf_tokenizer,
    max_length = max_length,
    padding = padding,
    truncation = truncation,
    is_split_into_words = is_split_into_words,
    n_tok_inps = as.integer(n_tok_inps),
    ...
  )

  if(!is.null(args[['max_length']]))
    args[['max_length']] <- as.integer(args[['max_length']])


  do.call(blurr()$data$all$HF_BeforeBatchTransform, args)

}



#' @title HF_CausalLMBeforeBatchTransform
#'
#' @description Handles everything you need to assemble a mini-batch of inputs and targets,
#' as well as decode the dictionary produced
#'
#' @details as a byproduct of the tokenization process in the `encodes` method.
#'
#' @param hf_arch architecture
#' @param hf_tokenizer tokenizer
#' @param max_length maximum length
#' @param padding padding or not
#' @param truncation truncation or not
#' @param is_split_into_words to split into words
#' @param n_tok_inps number tok inputs
#' @param ignore_token_id ignore token id
#' @param ... additional arguments
#' @return None
#' @export
HF_CausalLMBeforeBatchTransform <- function(hf_arch, hf_tokenizer, max_length = NULL,
                                            padding = TRUE, truncation = TRUE, is_split_into_words = FALSE,
                                            n_tok_inps = 1, ignore_token_id = -100, ...) {

  args <- list(
    hf_arch = hf_arch,
    hf_tokenizer = hf_tokenizer,
    max_length = max_length,
    padding = padding,
    truncation = truncation,
    is_split_into_words = is_split_into_words,
    n_tok_inps = as.integer(n_tok_inps),
    ignore_token_id = as.integer(ignore_token_id),
    ...
  )


  if(!is.null(args[['max_length']]))
    args[['max_length']] <- as.integer(args[['max_length']])

  do.call(blurr()$data$all$HF_CausalLMBeforeBatchTransform, args)

}


#' @title HF_SummarizationBeforeBatchTransform
#'
#' @description Handles everything you need to assemble a mini-batch of inputs and targets,
#' as well as decode the dictionary produced
#'
#' @details as a byproduct of the tokenization process in the `encodes` method.
#'
#' @param hf_arch architecture
#' @param hf_tokenizer tokenizer
#' @param max_length maximum length
#' @param padding padding or not
#' @param truncation truncation or not
#' @param is_split_into_words to split into words
#' @param n_tok_inps number tok inputs
#' @param ignore_token_id ignore token id
#' @param ... additional arguments
#'
#'
#' @return None
#' @export
HF_SummarizationBeforeBatchTransform <- function(hf_arch, hf_tokenizer, max_length = NULL,
                                                 padding = TRUE, truncation = TRUE,
                                                 is_split_into_words = FALSE, n_tok_inps = 2,
                                                 ignore_token_id = -100, ...) {

  args <- list(
    hf_arch = hf_arch,
    hf_tokenizer = hf_tokenizer,
    max_length = max_length,
    padding = padding,
    truncation = truncation,
    is_split_into_words = is_split_into_words,
    n_tok_inps = as.integer(n_tok_inps),
    ignore_token_id = as.integer(ignore_token_id),
    ...
  )

  if(!is.null(args[['max_length']]))
    args[['max_length']] <- as.integer(args[['max_length']])

  do.call(blurr()$data$all$HF_SummarizationBeforeBatchTransform, args)

}


#' @title HF_SummarizationInput
#' @return None
#'
#' @export
HF_SummarizationInput <- function() {

  blurr()$data$all$HF_SummarizationInput

}


#' @title HF_TASKS_ALL
#'
#' @description An enumeration.
#' @return None
#'
#' @export
HF_TASKS_ALL <- function() {

  blurr()$data$all$HF_TASKS_ALL

}


#' @title HF_TASKS_AUTO
#'
#' @description An enumeration.
#' @return None
#'
#' @export
HF_TASKS_AUTO <- function() {

  blurr()$data$all$HF_TASKS_AUTO

}


#' @title HF_Text2TextAfterBatchTransform
#'
#' @description Delegates (`__call__`,`decode`,`setup`) to (<code>encodes</code>,<code>decodes</code>,<code>setups</code>) if `split_idx` matches
#'
#'
#' @param hf_tokenizer tokenizer
#' @param input_return_type input return type
#' @return None
#' @export
HF_Text2TextAfterBatchTransform <- function(hf_tokenizer, input_return_type = HF_BaseInput()) {

  blurr()$data$all$HF_Text2TextAfterBatchTransform(
    hf_tokenizer = hf_tokenizer,
    input_return_type = input_return_type
  )

}


#' @title HF_Text2TextBlock
#'
#' @description A basic wrapper that links defaults transforms for the data block API
#'
#' @param ... parameters to pass
#' @return None
#'
#' @export
HF_Text2TextBlock <- function(...) {

  args = list(...)

  if(!is.null(args[['max_length']]))
    args[['max_length']] <- as.integer(args[['max_length']])

  if(!is.null(args[['n_tok_inps']]))
    args[['n_tok_inps']] <- as.integer(args[['n_tok_inps']])

  if(!is.null(args[['ignore_token_id']]))
    args[['ignore_token_id']] <- as.integer(args[['ignore_token_id']])

  do.call(blurr()$data$all$HF_Text2TextBlock, args)

}


#' @title HF_TokenCategorize
#'
#' @description Reversible transform of a list of category string to `vocab` id
#'
#'
#' @param vocab vocabulary
#' @param ignore_token ignore token
#' @param ignore_token_id ignore token id
#' @return None
#' @export
HF_TokenCategorize <- function(vocab = NULL, ignore_token = NULL, ignore_token_id = NULL) {

  args <- list(
    vocab = vocab,
    ignore_token = ignore_token,
    ignore_token_id = ignore_token_id
  )

  if(!is.null(args[['ignore_token_id']]))
    args[['ignore_token_id']] <- as.integer(args[['ignore_token_id']])


  do.call(blurr()$data$all$HF_TokenCategorize, args)

}



#' @title HF_TokenCategoryBlock
#'
#' @description `TransformBlock` for single-label categorical targets
#'
#' @param vocab vocabulary
#' @param ignore_token ignore token
#' @param ignore_token_id ignore token id
#' @return None
#' @export
HF_TokenCategoryBlock <- function(vocab = NULL, ignore_token = NULL, ignore_token_id = NULL) {

  args <- list(
    vocab = vocab,
    ignore_token = ignore_token,
    ignore_token_id = ignore_token_id
  )

  if(!is.null(args[['ignore_token_id']]))
    args[['ignore_token_id']] <- as.integer(args[['ignore_token_id']])

  do.call(blurr()$data$all$HF_TokenCategoryBlock, args)

}


#' @title HF_TokenClassBeforeBatchTransform
#'
#' @description Handles everything you need to assemble a mini-batch of inputs and targets,
#' as well as decode the dictionary produced
#'
#' @details as a byproduct of the tokenization process in the `encodes` method.
#'
#' @param hf_arch architecture
#' @param hf_tokenizer tokenizer
#' @param ignore_token_id ignore token id
#' @param max_length maximum length
#' @param padding padding or not
#' @param truncation truncation or not
#' @param is_split_into_words to split into_words
#' @param n_tok_inps number tok inputs
#' @param ... additional arguments
#' @return None
#' @export
HF_TokenClassBeforeBatchTransform <- function(hf_arch, hf_tokenizer, ignore_token_id = -100, max_length = NULL,
                                              padding = TRUE, truncation = TRUE, is_split_into_words = TRUE,
                                              n_tok_inps = 1, ...) {

  args <- list(
    hf_arch = hf_arch,
    hf_tokenizer = hf_tokenizer,
    ignore_token_id = ignore_token_id,
    max_length = max_length,
    padding = padding,
    truncation = truncation,
    is_split_into_words = is_split_into_words,
    n_tok_inps = n_tok_inps,
    ...
  )

  if(!is.null(args[['max_length']]))
    args[['max_length']] <- as.integer(args[['max_length']])

  if(!is.null(args[['n_tok_inps']]))
    args[['n_tok_inps']] <- as.integer(args[['n_tok_inps']])

  if(!is.null(args[['ignore_token_id']]))
    args[['ignore_token_id']] <- as.integer(args[['ignore_token_id']])

  do.call(blurr()$data$all$HF_TokenClassBeforeBatchTransform, args)

}


#' @title HF_TokenClassInput
#' @return None
#'
#' @export
HF_TokenClassInput <- function() {

  blurr()$data$all$HF_TokenClassInput

}

#' @title HF_TokenTensorCategory
#' @return None
#'
#' @export
HF_TokenTensorCategory <- function() {

  blurr()$data$all$HF_TokenTensorCategory

}



#' @title Calculate_rouge
#'
#'
#' @param predicted_txts predicted texts
#' @param reference_txts reference texts
#' @param rouge_keys rouge keys
#' @param use_stemmer use stemmer or not
#' @return None
#'
#' @export
calculate_rouge <- function(predicted_txts, reference_txts,
                            rouge_keys = c("rouge1", "rouge2", "rougeL"),
                            use_stemmer = TRUE) {


  if(missing(predicted_txts) & missing(reference_txts)) {
    blurr()$modeling$all$calculate_rouge
  } else {
    args <- list(
      predicted_txts = predicted_txts,
      reference_txts = reference_txts,
      rouge_keys = rouge_keys,
      use_stemmer = use_stemmer
    )

    do.call(blurr()$modeling$all$calculate_rouge, args)
  }

}

#' @title HF_SummarizationModelCallback
#'
#' @description Basic class handling tweaks of the training loop by changing a `Learner` in various events
#'
#'
#' @param rouge_metrics rouge metrics
#' @param ignore_token_id integer, ignore token id
#' @param ... additional arguments
#' @return None
#' @export
HF_SummarizationModelCallback <- function(rouge_metrics = c("rouge1", "rouge2", "rougeL"),
                                          ignore_token_id = -100, ...) {

  args <- list(
    rouge_metrics = rouge_metrics,
    ignore_token_id = as.integer(ignore_token_id),
    ...
  )


  do.call(blurr()$modeling$all$HF_SummarizationModelCallback, args)

}

#' @title Summarization_splitter
#'
#' @description Custom param splitter for summarization models
#'
#'
#' @param m splitter parameter
#' @param arch architecture
#' @return None
#' @export
summarization_splitter <- function(m, arch) {

  if(missing(m) & missing(arch)) {
    blurr()$modeling$all$summarization_splitter
  } else {
    blurr()$modeling$all$summarization_splitter(
      m = m,
      arch = arch
    )
  }

}

