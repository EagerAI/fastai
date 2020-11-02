## ----setup, include=FALSE-----------------------------------------------------
knitr::opts_chunk$set(echo = TRUE, eval = FALSE)

## -----------------------------------------------------------------------------
#  library(fastai)
#  library(magrittr)
#  
#  URLs_WIKITEXT()
#  
#  path = 'wikitext-2'
#  
#  train = data.table::fread(paste(path, 'train.csv', sep = '/'), header = FALSE, fill = TRUE)
#  
#  test = data.table::fread(paste(path, 'test.csv', sep = '/'), header = FALSE, fill = TRUE)
#  
#  df = rbind(train, test)
#  
#  rm(train,test)

## -----------------------------------------------------------------------------
#  tr = reticulate::import('transformers')
#  pretrained_weights = 'gpt2'
#  tokenizer = tr$GPT2TokenizerFast$from_pretrained(pretrained_weights)
#  model = tr$GPT2LMHeadModel$from_pretrained(pretrained_weights)

## -----------------------------------------------------------------------------
#  tls = TfmdLists(tokenized, TransformersTokenizer(tokenizer),
#                                   splits = splits,
#                                   dl_type = LMDataLoader())
#  
#  # Now, we are ready to create our Learner, which is a fastai object grouping data, model
#  # and loss function and handles model training or inference. Since we are in a language
#  #model setting, we pass perplexity as a metric, and we need to use the callback we just
#  # defined. Lastly, we use mixed precision to save every bit of memory we can (and if you
#  # have a modern GPU, it will also make training faster):
#  learn = Learner(dls, model, loss_func=CrossEntropyLossFlat(),
#                  cbs = list(TransformersDropOutput()),
#                  metrics = Perplexity())$to_fp16()
#  
#  learn %>% fit_one_cycle(1, 1e-4)

## -----------------------------------------------------------------------------
#  prompt = "\n = Unicorn = \n \n A unicorn is a magical creature with a rainbow tail and a horn"
#  prompt_ids = tokenizer$encode(prompt)
#  inp = tensor(prompt_ids)[NULL]$cuda()
#  preds = learn$model$generate(inp, max_length=80L, num_beams=5L, temperature=1.5)
#  tokenizer$decode(as.integer(preds[0]$cpu()$numpy()))

