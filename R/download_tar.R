
#' @title ADULT_SAMPLE dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download ADULT_SAMPLE dataset
#' @export
URLs_ADULT_SAMPLE <- function(filename = ADULT_SAMPLE, untar = TRUE) {

  download.file(paste(tabular$URLs$ADULT_SAMPLE,'.tgz',sep = ''),
                destfile = paste(filename,'tgz',sep = ''))

  if(untar)
    untar(filename)

}


#' @title AG_NEWS dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download AG_NEWS dataset
#' @export
URLs_AG_NEWS <- function(filename = AG_NEWS, untar = TRUE) {

  download.file(paste(tabular$URLs$AG_NEWS,'.tgz',sep = ''),
                destfile = paste(filename,'tgz',sep = ''))

  if(untar)
    untar(filename)

}


#' @title AMAZON_REVIEWSAMAZON_REVIEWS dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download AMAZON_REVIEWSAMAZON_REVIEWS dataset
#' @export
URLs_AMAZON_REVIEWSAMAZON_REVIEWS <- function(filename = AMAZON_REVIEWSAMAZON_REVIEWS, untar = TRUE) {

  download.file(paste(tabular$URLs$AMAZON_REVIEWSAMAZON_REVIEWS,'.tgz',sep = ''),
                destfile = paste(filename,'tgz',sep = ''))

  if(untar)
    untar(filename)

}


#' @title AMAZON_REVIEWS_POLARITY dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download AMAZON_REVIEWS_POLARITY dataset
#' @export
URLs_AMAZON_REVIEWS_POLARITY <- function(filename = AMAZON_REVIEWS_POLARITY, untar = TRUE) {

  download.file(paste(tabular$URLs$AMAZON_REVIEWS_POLARITY,'.tgz',sep = ''),
                destfile = paste(filename,'tgz',sep = ''))

  if(untar)
    untar(filename)

}


#' @title BIWI_HEAD_POSE dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download BIWI_HEAD_POSE dataset
#' @export
URLs_BIWI_HEAD_POSE <- function(filename = BIWI_HEAD_POSE, untar = TRUE) {

  download.file(paste(tabular$URLs$BIWI_HEAD_POSE,'.tgz',sep = ''),
                destfile = paste(filename,'tgz',sep = ''))

  if(untar)
    untar(filename)

}


#' @title CALTECH_101 dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download CALTECH_101 dataset
#' @export
URLs_CALTECH_101 <- function(filename = CALTECH_101, untar = TRUE) {

  download.file(paste(tabular$URLs$CALTECH_101,'.tgz',sep = ''),
                destfile = paste(filename,'tgz',sep = ''))

  if(untar)
    untar(filename)

}


#' @title CAMVID dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download CAMVID dataset
#' @export
URLs_CAMVID <- function(filename = CAMVID, untar = TRUE) {

  download.file(paste(tabular$URLs$CAMVID,'.tgz',sep = ''),
                destfile = paste(filename,'tgz',sep = ''))

  if(untar)
    untar(filename)

}


#' @title CAMVID_TINY dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download CAMVID_TINY dataset
#' @export
URLs_CAMVID_TINY <- function(filename = CAMVID_TINY, untar = TRUE) {

  download.file(paste(tabular$URLs$CAMVID_TINY,'.tgz',sep = ''),
                destfile = paste(filename,'tgz',sep = ''))

  if(untar)
    untar(filename)

}


#' @title CARS dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download CARS dataset
#' @export
URLs_CARS <- function(filename = CARS, untar = TRUE) {

  download.file(paste(tabular$URLs$CARS,'.tgz',sep = ''),
                destfile = paste(filename,'tgz',sep = ''))

  if(untar)
    untar(filename)

}


#' @title CIFAR dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download CIFAR dataset
#' @export
URLs_CIFAR <- function(filename = CIFAR, untar = TRUE) {

  download.file(paste(tabular$URLs$CIFAR,'.tgz',sep = ''),
                destfile = paste(filename,'tgz',sep = ''))

  if(untar)
    untar(filename)

}


#' @title CIFAR_100 dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download CIFAR_100 dataset
#' @export
URLs_CIFAR_100 <- function(filename = CIFAR_100, untar = TRUE) {

  download.file(paste(tabular$URLs$CIFAR_100,'.tgz',sep = ''),
                destfile = paste(filename,'tgz',sep = ''))

  if(untar)
    untar(filename)

}


#' @title COCO_TINY dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download COCO_TINY dataset
#' @export
URLs_COCO_TINY <- function(filename = COCO_TINY, untar = TRUE) {

  download.file(paste(tabular$URLs$COCO_TINY,'.tgz',sep = ''),
                destfile = paste(filename,'tgz',sep = ''))

  if(untar)
    untar(filename)

}


#' @title CUB_200_2011 dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download CUB_200_2011 dataset
#' @export
URLs_CUB_200_2011 <- function(filename = CUB_200_2011, untar = TRUE) {

  download.file(paste(tabular$URLs$CUB_200_2011,'.tgz',sep = ''),
                destfile = paste(filename,'tgz',sep = ''))

  if(untar)
    untar(filename)

}


#' @title DBPEDIA dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download DBPEDIA dataset
#' @export
URLs_DBPEDIA <- function(filename = DBPEDIA, untar = TRUE) {

  download.file(paste(tabular$URLs$DBPEDIA,'.tgz',sep = ''),
                destfile = paste(filename,'tgz',sep = ''))

  if(untar)
    untar(filename)

}


#' @title DOGS dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download DOGS dataset
#' @export
URLs_DOGS <- function(filename = DOGS, untar = TRUE) {

  download.file(paste(tabular$URLs$DOGS,'.tgz',sep = ''),
                destfile = paste(filename,'tgz',sep = ''))

  if(untar)
    untar(filename)

}


#' @title FLOWERS dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download FLOWERS dataset
#' @export
URLs_FLOWERS <- function(filename = FLOWERS, untar = TRUE) {

  download.file(paste(tabular$URLs$FLOWERS,'.tgz',sep = ''),
                destfile = paste(filename,'tgz',sep = ''))

  if(untar)
    untar(filename)

}


#' @title FOOD dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download FOOD dataset
#' @export
URLs_FOOD <- function(filename = FOOD, untar = TRUE) {

  download.file(paste(tabular$URLs$FOOD,'.tgz',sep = ''),
                destfile = paste(filename,'tgz',sep = ''))

  if(untar)
    untar(filename)

}


#' @title HUMAN_NUMBERS dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download HUMAN_NUMBERS dataset
#' @export
URLs_HUMAN_NUMBERS <- function(filename = HUMAN_NUMBERS, untar = TRUE) {

  download.file(paste(tabular$URLs$HUMAN_NUMBERS,'.tgz',sep = ''),
                destfile = paste(filename,'tgz',sep = ''))

  if(untar)
    untar(filename)

}


#' @title IMAGENETTE dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download IMAGENETTE dataset
#' @export
URLs_IMAGENETTE <- function(filename = IMAGENETTE, untar = TRUE) {

  download.file(paste(tabular$URLs$IMAGENETTE,'.tgz',sep = ''),
                destfile = paste(filename,'tgz',sep = ''))

  if(untar)
    untar(filename)

}


#' @title IMAGENETTE_160 dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download IMAGENETTE_160 dataset
#' @export
URLs_IMAGENETTE_160 <- function(filename = IMAGENETTE_160, untar = TRUE) {

  download.file(paste(tabular$URLs$IMAGENETTE_160,'.tgz',sep = ''),
                destfile = paste(filename,'tgz',sep = ''))

  if(untar)
    untar(filename)

}


#' @title IMAGENETTE_320 dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download IMAGENETTE_320 dataset
#' @export
URLs_IMAGENETTE_320 <- function(filename = IMAGENETTE_320, untar = TRUE) {

  download.file(paste(tabular$URLs$IMAGENETTE_320,'.tgz',sep = ''),
                destfile = paste(filename,'tgz',sep = ''))

  if(untar)
    untar(filename)

}


#' @title IMAGEWOOF dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download IMAGEWOOF dataset
#' @export
URLs_IMAGEWOOF <- function(filename = IMAGEWOOF, untar = TRUE) {

  download.file(paste(tabular$URLs$IMAGEWOOF,'.tgz',sep = ''),
                destfile = paste(filename,'tgz',sep = ''))

  if(untar)
    untar(filename)

}


#' @title IMAGEWOOF_160 dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download IMAGEWOOF_160 dataset
#' @export
URLs_IMAGEWOOF_160 <- function(filename = IMAGEWOOF_160, untar = TRUE) {

  download.file(paste(tabular$URLs$IMAGEWOOF_160,'.tgz',sep = ''),
                destfile = paste(filename,'tgz',sep = ''))

  if(untar)
    untar(filename)

}


#' @title IMAGEWOOF_320 dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download IMAGEWOOF_320 dataset
#' @export
URLs_IMAGEWOOF_320 <- function(filename = IMAGEWOOF_320, untar = TRUE) {

  download.file(paste(tabular$URLs$IMAGEWOOF_320,'.tgz',sep = ''),
                destfile = paste(filename,'tgz',sep = ''))

  if(untar)
    untar(filename)

}


#' @title IMDB dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download IMDB dataset
#' @export
URLs_IMDB <- function(filename = IMDB, untar = TRUE) {

  download.file(paste(tabular$URLs$IMDB,'.tgz',sep = ''),
                destfile = paste(filename,'tgz',sep = ''))

  if(untar)
    untar(filename)

}


#' @title IMDB_SAMPLE dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download IMDB_SAMPLE dataset
#' @export
URLs_IMDB_SAMPLE <- function(filename = IMDB_SAMPLE, untar = TRUE) {

  download.file(paste(tabular$URLs$IMDB_SAMPLE,'.tgz',sep = ''),
                destfile = paste(filename,'tgz',sep = ''))

  if(untar)
    untar(filename)

}


#' @title LSUN_BEDROOMS dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download LSUN_BEDROOMS dataset
#' @export
URLs_LSUN_BEDROOMS <- function(filename = LSUN_BEDROOMS, untar = TRUE) {

  download.file(paste(tabular$URLs$LSUN_BEDROOMS,'.tgz',sep = ''),
                destfile = paste(filename,'tgz',sep = ''))

  if(untar)
    untar(filename)

}


#' @title ML_SAMPLE dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download ML_SAMPLE dataset
#' @export
URLs_ML_SAMPLE <- function(filename = ML_SAMPLE, untar = TRUE) {

  download.file(paste(tabular$URLs$ML_SAMPLE,'.tgz',sep = ''),
                destfile = paste(filename,'tgz',sep = ''))

  if(untar)
    untar(filename)

}


#' @title MNIST dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download MNIST dataset
#' @export
URLs_MNIST <- function(filename = MNIST, untar = TRUE) {

  download.file(paste(tabular$URLs$MNIST,'.tgz',sep = ''),
                destfile = paste(filename,'tgz',sep = ''))

  if(untar)
    untar(filename)

}


#' @title MNIST_SAMPLE dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download MNIST_SAMPLE dataset
#' @export
URLs_MNIST_SAMPLE <- function(filename = MNIST_SAMPLE, untar = TRUE) {

  download.file(paste(tabular$URLs$MNIST_SAMPLE,'.tgz',sep = ''),
                destfile = paste(filename,'tgz',sep = ''))

  if(untar)
    untar(filename)

}


#' @title MNIST_TINY dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download MNIST_TINY dataset
#' @export
URLs_MNIST_TINY <- function(filename = MNIST_TINY, untar = TRUE) {

  download.file(paste(tabular$URLs$MNIST_TINY,'.tgz',sep = ''),
                destfile = paste(filename,'tgz',sep = ''))

  if(untar)
    untar(filename)

}


#' @title MNIST_VAR_SIZE_TINY dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download MNIST_VAR_SIZE_TINY dataset
#' @export
URLs_MNIST_VAR_SIZE_TINY <- function(filename = MNIST_VAR_SIZE_TINY, untar = TRUE) {

  download.file(paste(tabular$URLs$MNIST_VAR_SIZE_TINY,'.tgz',sep = ''),
                destfile = paste(filename,'tgz',sep = ''))

  if(untar)
    untar(filename)

}


#' @title MT_ENG_FRA dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download MT_ENG_FRA dataset
#' @export
URLs_MT_ENG_FRA <- function(filename = MT_ENG_FRA, untar = TRUE) {

  download.file(paste(tabular$URLs$MT_ENG_FRA,'.tgz',sep = ''),
                destfile = paste(filename,'tgz',sep = ''))

  if(untar)
    untar(filename)

}


#' @title OPENAI_TRANSFORMER dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download OPENAI_TRANSFORMER dataset
#' @export
URLs_OPENAI_TRANSFORMER <- function(filename = OPENAI_TRANSFORMER, untar = TRUE) {

  download.file(paste(tabular$URLs$OPENAI_TRANSFORMER,'.tgz',sep = ''),
                destfile = paste(filename,'tgz',sep = ''))

  if(untar)
    untar(filename)

}


#' @title PASCAL_2007 dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download PASCAL_2007 dataset
#' @export
URLs_PASCAL_2007 <- function(filename = PASCAL_2007, untar = TRUE) {

  download.file(paste(tabular$URLs$PASCAL_2007,'.tgz',sep = ''),
                destfile = paste(filename,'tgz',sep = ''))

  if(untar)
    untar(filename)

}


#' @title PASCAL_2012 dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download PASCAL_2012 dataset
#' @export
URLs_PASCAL_2012 <- function(filename = PASCAL_2012, untar = TRUE) {

  download.file(paste(tabular$URLs$PASCAL_2012,'.tgz',sep = ''),
                destfile = paste(filename,'tgz',sep = ''))

  if(untar)
    untar(filename)

}


#' @title PETS dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download PETS dataset
#' @export
URLs_PETS <- function(filename = PETS, untar = TRUE) {

  download.file(paste(tabular$URLs$PETS,'.tgz',sep = ''),
                destfile = paste(filename,'tgz',sep = ''))

  if(untar)
    untar(filename)

}


#' @title PLANET_SAMPLE dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download PLANET_SAMPLE dataset
#' @export
URLs_PLANET_SAMPLE <- function(filename = PLANET_SAMPLE, untar = TRUE) {

  download.file(paste(tabular$URLs$PLANET_SAMPLE,'.tgz',sep = ''),
                destfile = paste(filename,'tgz',sep = ''))

  if(untar)
    untar(filename)

}


#' @title PLANET_TINY dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download PLANET_TINY dataset
#' @export
URLs_PLANET_TINY <- function(filename = PLANET_TINY, untar = TRUE) {

  download.file(paste(tabular$URLs$PLANET_TINY,'.tgz',sep = ''),
                destfile = paste(filename,'tgz',sep = ''))

  if(untar)
    untar(filename)

}


#' @title S3_COCO dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download S3_COCO dataset
#' @export
URLs_S3_COCO <- function(filename = S3_COCO, untar = TRUE) {

  download.file(paste(tabular$URLs$S3_COCO,'.tgz',sep = ''),
                destfile = paste(filename,'tgz',sep = ''))

  if(untar)
    untar(filename)

}


#' @title S3_IMAGE dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download S3_IMAGE dataset
#' @export
URLs_S3_IMAGE <- function(filename = S3_IMAGE, untar = TRUE) {

  download.file(paste(tabular$URLs$S3_IMAGE,'.tgz',sep = ''),
                destfile = paste(filename,'tgz',sep = ''))

  if(untar)
    untar(filename)

}


#' @title S3_IMAGELOC dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download S3_IMAGELOC dataset
#' @export
URLs_S3_IMAGELOC <- function(filename = S3_IMAGELOC, untar = TRUE) {

  download.file(paste(tabular$URLs$S3_IMAGELOC,'.tgz',sep = ''),
                destfile = paste(filename,'tgz',sep = ''))

  if(untar)
    untar(filename)

}


#' @title S3_MODEL dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download S3_MODEL dataset
#' @export
URLs_S3_MODEL <- function(filename = S3_MODEL, untar = TRUE) {

  download.file(paste(tabular$URLs$S3_MODEL,'.tgz',sep = ''),
                destfile = paste(filename,'tgz',sep = ''))

  if(untar)
    untar(filename)

}


#' @title S3_NLP dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download S3_NLP dataset
#' @export
URLs_S3_NLP <- function(filename = S3_NLP, untar = TRUE) {

  download.file(paste(tabular$URLs$S3_NLP,'.tgz',sep = ''),
                destfile = paste(filename,'tgz',sep = ''))

  if(untar)
    untar(filename)

}


#' @title SKIN_LESION dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download SKIN_LESION dataset
#' @export
URLs_SKIN_LESION <- function(filename = SKIN_LESION, untar = TRUE) {

  download.file(paste(tabular$URLs$SKIN_LESION,'.tgz',sep = ''),
                destfile = paste(filename,'tgz',sep = ''))

  if(untar)
    untar(filename)

}


#' @title SOGOU_NEWS dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download SOGOU_NEWS dataset
#' @export
URLs_SOGOU_NEWS <- function(filename = SOGOU_NEWS, untar = TRUE) {

  download.file(paste(tabular$URLs$SOGOU_NEWS,'.tgz',sep = ''),
                destfile = paste(filename,'tgz',sep = ''))

  if(untar)
    untar(filename)

}


#' @title WIKITEXT dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download WIKITEXT dataset
#' @export
URLs_WIKITEXT <- function(filename = WIKITEXT, untar = TRUE) {

  download.file(paste(tabular$URLs$WIKITEXT,'.tgz',sep = ''),
                destfile = paste(filename,'tgz',sep = ''))

  if(untar)
    untar(filename)

}


#' @title WIKITEXT_TINY dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download WIKITEXT_TINY dataset
#' @export
URLs_WIKITEXT_TINY <- function(filename = WIKITEXT_TINY, untar = TRUE) {

  download.file(paste(tabular$URLs$WIKITEXT_TINY,'.tgz',sep = ''),
                destfile = paste(filename,'tgz',sep = ''))

  if(untar)
    untar(filename)

}


#' @title WT103_BWD dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download WT103_BWD dataset
#' @export
URLs_WT103_BWD <- function(filename = WT103_BWD, untar = TRUE) {

  download.file(paste(tabular$URLs$WT103_BWD,'.tgz',sep = ''),
                destfile = paste(filename,'tgz',sep = ''))

  if(untar)
    untar(filename)

}


#' @title WT103_FWD dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download WT103_FWD dataset
#' @export
URLs_WT103_FWD <- function(filename = WT103_FWD, untar = TRUE) {

  download.file(paste(tabular$URLs$WT103_FWD,'.tgz',sep = ''),
                destfile = paste(filename,'tgz',sep = ''))

  if(untar)
    untar(filename)

}


#' @title YAHOO_ANSWERS dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download YAHOO_ANSWERS dataset
#' @export
URLs_YAHOO_ANSWERS <- function(filename = YAHOO_ANSWERS, untar = TRUE) {

  download.file(paste(tabular$URLs$YAHOO_ANSWERS,'.tgz',sep = ''),
                destfile = paste(filename,'tgz',sep = ''))

  if(untar)
    untar(filename)

}


#' @title YELP_REVIEWS dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download YELP_REVIEWS dataset
#' @export
URLs_YELP_REVIEWS <- function(filename = YELP_REVIEWS, untar = TRUE) {

  download.file(paste(tabular$URLs$YELP_REVIEWS,'.tgz',sep = ''),
                destfile = paste(filename,'tgz',sep = ''))

  if(untar)
    untar(filename)

}


#' @title YELP_REVIEWS_POLARITY dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download YELP_REVIEWS_POLARITY dataset
#' @export
URLs_YELP_REVIEWS_POLARITY <- function(filename = YELP_REVIEWS_POLARITY, untar = TRUE) {

  download.file(paste(tabular$URLs$YELP_REVIEWS_POLARITY,'.tgz',sep = ''),
                destfile = paste(filename,'tgz',sep = ''))

  if(untar)
    untar(filename)

}


