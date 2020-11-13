
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


#' @title HF_TASKS_AUTO
#'
#' @description An enumeration.
#'
#' @return None
#' @export
HF_TASKS_AUTO <- function() {

  blurr()$data$all$HF_TASKS_AUTO

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


#' @title HF_TASKS_ALL
#'
#' @description An enumeration.
#'
#' @return None
#' @export
HF_TASKS_ALL <- function() {

  blurr()$data$all$HF_TASKS_ALL

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





