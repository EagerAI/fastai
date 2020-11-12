
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
#' @param hf_batch_tfm batch transformation
#' @param max_length maximum length
#' @param padding padding or not
#' @param truncation truncation or not
#' @param is_split_into_words to split into words or not
#' @param n_tok_inps number of tok inps
#' @param tok_kwargs tokenizer additional arguments
#' @param hf_input_return_type input return type
#' @param dl_type dataloader type
#' @param batch_kwargs batch additional arguments
#' @return None
#' @export
HF_TextBlock <- function(hf_arch=NULL, hf_tokenizer=NULL, before_batch_tfms=NULL, after_batch_tfms=NULL,
                         max_length=512, padding=TRUE, truncation=TRUE, is_split_into_words=FALSE,
                         n_tok_inps=1, tok_kwargs=list(), input_return_type=HF_BaseInput(), dl_type=SortedDL(),
                         before_batch_kwargs=list(), after_batch_kwargs=list(), ...) {

  args <- list(
    hf_arch=hf_arch,
    hf_tokenizer=hf_tokenizer,
    before_batch_tfms=before_batch_tfms,
    after_batch_tfms=after_batch_tfms,
    max_length=as.integer(max_length),
    padding=padding,
    truncation=truncation,
    is_split_into_words=is_split_into_words,
    n_tok_inps=n_tok_inps,
    tok_kwargs=tok_kwargs,
    input_return_type=input_return_type,
    dl_type=dl_type,
    before_batch_kwargs=before_batch_kwargs,
    after_batch_kwargs=after_batch_kwargs,
    ...
  )

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
#'
#' @export
get_hf_objects <- function(...) {

  args <- list(
    ...
  )

  do.call(blurr()$data$all$BLURR_MODEL_HELPER$get_hf_objects, args)

}






