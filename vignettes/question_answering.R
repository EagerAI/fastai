## ----setup, include=FALSE-----------------------------------------------------
knitr::opts_chunk$set(echo = TRUE, eval = FALSE)

## -----------------------------------------------------------------------------
#  reticulate::py_install('ohmeow-blurr',pip = TRUE)

## -----------------------------------------------------------------------------
#  library(fastai)
#  library(magrittr)
#  library(zeallot)
#  squad_df = data.table::fread('https://raw.githubusercontent.com/ohmeow/blurr/master/nbs/squad_sample.csv')

## -----------------------------------------------------------------------------
#  pretrained_model_name = 'bert-large-uncased-whole-word-masking-finetuned-squad'
#  
#  hf_model_cls = transformers$BertForQuestionAnswering
#  
#  c(hf_arch, hf_config, hf_tokenizer, hf_model) %<-%
#    get_hf_objects(pretrained_model_name, model_cls=hf_model_cls)
#  
#  preprocess = partial(pre_process_squad(), hf_arch=hf_arch, hf_tokenizer=hf_tokenizer)

## -----------------------------------------------------------------------------
#  squad_df = data.table::as.data.table(squad_df %>% py_apply(preprocess))
#  max_seq_len = 128
#  
#  tibble::tibble(squad_df)
#  
#  squad_df[, c(8,10:12)] = lapply(squad_df[, c(8,10:12)], function(x) unlist(as.vector(x)))
#  squad_df = squad_df[is_impossible == FALSE & tokenized_input_len < max_seq_len]
#  vocab = c(1:max_seq_len)

## -----------------------------------------------------------------------------
#  trunc_strat = ifelse(hf_tokenizer$padding_side == 'right', 'only_second', 'only_first')
#  
#  before_batch_tfm = HF_QABeforeBatchTransform(hf_arch, hf_tokenizer,
#                                               max_length = max_seq_len,
#                                               truncation = trunc_strat,
#                                               tok_kwargs = list('return_special_tokens_mask' = TRUE))
#  
#  blocks = list(
#    HF_TextBlock(before_batch_tfms=before_batch_tfm, input_return_type=HF_QuestionAnswerInput),
#    CategoryBlock(vocab=vocab),
#    CategoryBlock(vocab=vocab)
#  )
#  
#  # question and context
#  get_x = function(x) {
#    if(hf_tokenizer$padding_side == 'right') {
#      list(x[['question']], x[['context']])
#    } else {
#      list(x[['context']], x[['question']])
#    }
#  }
#  
#  dblock = DataBlock(blocks=blocks,
#                     get_x=get_x,
#                     get_y=list(ColReader('tok_answer_start'), ColReader('tok_answer_end')),
#                     splitter=RandomSplitter(),
#                     n_inp=1)
#  
#  dls = dblock %>% dataloaders(squad_df, bs=4)
#  
#  dls %>% one_batch()

## -----------------------------------------------------------------------------
#  
#  model = HF_BaseModelWrapper(hf_model)
#  
#  learn = Learner(dls,
#                  model,
#                  opt_func=partial(Adam, decouple_wd=T),
#                  cbs=HF_QstAndAnsModelCallback(),
#                  splitter=hf_splitter())
#  
#  learn$loss_func=MultiTargetLoss()
#  learn$create_opt()                # -> will create your layer groups based on your "splitter" function
#  learn$freeze()
#  
#  learn %>% fit_one_cycle(4, lr_max=1e-3)

## -----------------------------------------------------------------------------
#  inf_df = data.frame( 'question'= 'When was Star Wars made?',
#                       'context'= 'George Lucas created Star Wars in 1977. He directed and produced it.')

## -----------------------------------------------------------------------------
#  bert_answer = function(inf_df) {
#    test_dl = dls$test_dl(inf_df)
#    inp = test_dl$one_batch()[[1]]['input_ids']
#  
#    res = learn %>% predict(inf_df)
#  
#    # as_array is a function to turn a torch tensor to R array
#    sapply(res[[3]],as_array)
#  
#    hf_tokenizer$convert_ids_to_tokens(inp[[1]]$tolist()[[1]],
#                                       skip_special_tokens=FALSE)[sapply(res[[3]],as_array)+1]
#    # [sapply(res[[3]],as_array)+1] here +1 because tensor starts from 0 but R from 1
#  }

## -----------------------------------------------------------------------------
#  cat(bert_answer(inf_df))
#  # in 1977

