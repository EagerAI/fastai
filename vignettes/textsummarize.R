## ----setup, include=FALSE-----------------------------------------------------
knitr::opts_chunk$set(echo = TRUE, eval = FALSE)

## -----------------------------------------------------------------------------
#  reticulate::py_install('ohmeow-blurr',pip = TRUE)

## -----------------------------------------------------------------------------
#  library(fastai)
#  library(magrittr)
#  library(zeallot)
#  
#  cnndm_df = data.table::fread('https://raw.githubusercontent.com/ohmeow/blurr/master/nbs/cnndm_sample.csv')

## -----------------------------------------------------------------------------
#  transformers = transformers()
#  
#  BartForConditionalGeneration = transformers$BartForConditionalGeneration
#  
#  pretrained_model_name = "facebook/bart-large-cnn"
#  c(hf_arch, hf_config, hf_tokenizer, hf_model) %<-%
#    get_hf_objects(pretrained_model_name,model_cls=BartForConditionalGeneration)

## -----------------------------------------------------------------------------
#  
#  before_batch_tfm = HF_SummarizationBeforeBatchTransform(hf_arch,
#                                                          hf_tokenizer, max_length=c(256, 130))
#  blocks = list(HF_Text2TextBlock(before_batch_tfms=before_batch_tfm,
#                                  input_return_type=HF_SummarizationInput), noop())
#  
#  dblock = DataBlock(blocks=blocks,
#                     get_x=ColReader('article'),
#                     get_y=ColReader('highlights'),
#                     splitter=RandomSplitter())
#  
#  dls = dblock %>% dataloaders(cnndm_df, bs=2)
#  
#  dls %>% one_batch()
#  

## -----------------------------------------------------------------------------
#  text_gen_kwargs = hf_config$task_specific_params['summarization'][[1]]
#  text_gen_kwargs['max_length'] = 130L; text_gen_kwargs['min_length'] = 30L
#  
#  text_gen_kwargs
#  
#  model = HF_BaseModelWrapper(hf_model)
#  model_cb = HF_SummarizationModelCallback(text_gen_kwargs=text_gen_kwargs)
#  
#  learn = Learner(dls,
#                  model,
#                  opt_func=partial(Adam),
#                  loss_func=CrossEntropyLossFlat(), #HF_PreCalculatedLoss()
#                  cbs=model_cb,
#                  splitter=partial(summarization_splitter, arch=hf_arch)) #.to_native_fp16() #.to_fp16()
#  
#  learn$create_opt()
#  learn$freeze()
#  
#  learn %>% fit_one_cycle(1, lr_max=4e-5)

## -----------------------------------------------------------------------------
#  test_article = c("About 10 men armed with pistols and small machine guns raided a casino in Switzerland
#  and made off into France with several hundred thousand Swiss francs in the early hours
#  of Sunday morning, police said. The men, dressed in black clothes and black ski masks,
#  split into two groups during the raid on the Grand Casino Basel, Chief Inspector Peter
#  Gill told CNN. One group tried to break into the casino's vault on the lower level
#  but could not get in, but they did rob the cashier of the money that was not secured,
#  he said. The second group of armed robbers entered the upper level where the roulette
#  and blackjack tables are located and robbed the cashier there, he said. As the thieves
#  were leaving the casino, a woman driving by and unaware of what was occurring unknowingly
#  blocked the armed robbers' vehicles. A gunman pulled the woman from her vehicle, beat
#  her, and took off for the French border. The other gunmen followed into France, which
#  is only about 100 meters (yards) from the casino, Gill said. There were about 600 people
#  in the casino at the time of the robbery. There were no serious injuries, although one
#  guest on the Casino floor was kicked in the head by one of the robbers when he moved,
#  the police officer said. Swiss authorities are working closely with French authorities,
#  Gill said. The robbers spoke French and drove vehicles with French lRicense plates.
#  CNN's Andreena Narayan contributed to this report.")

## -----------------------------------------------------------------------------
#  outputs = learn$blurr_summarize(test_article, num_return_sequences=3L)
#  cat(outputs)

