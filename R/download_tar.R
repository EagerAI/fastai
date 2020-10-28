
#' @title ADULT_SAMPLE dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download ADULT_SAMPLE dataset
#' @return None
#' @examples
#' \dontrun{
#'
#' URLs_ADULT_SAMPLE()
#'
#' }
#'
#' @export
URLs_ADULT_SAMPLE <- function(filename = 'ADULT_SAMPLE', untar = TRUE) {

  download.file(paste(tabular$URLs$ADULT_SAMPLE,sep = ''),
                destfile = paste(filename,'.tgz',sep = ''))

  if(untar)
    untar(paste(filename,'.tgz',sep = ''))

}


#' @title AG_NEWS dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download AG_NEWS dataset
#' @return None
#'
#' @examples
#' \dontrun{
#'
#' URLs_AG_NEWS()
#'
#' }
#'
#' @export
URLs_AG_NEWS <- function(filename = 'AG_NEWS', untar = TRUE) {

  download.file(paste(tabular$URLs$AG_NEWS, sep = ''),
                destfile = paste(filename,'.tgz',sep = ''))

  if(untar)
    untar(paste(filename,'.tgz',sep = ''))

}


#' @title AMAZON_REVIEWSAMAZON_REVIEWS dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download AMAZON_REVIEWSAMAZON_REVIEWS dataset
#' @return None
#' @export
URLs_AMAZON_REVIEWSAMAZON_REVIEWS <- function(filename = 'AMAZON_REVIEWSAMAZON_REVIEWS', untar = TRUE) {

  download.file(paste(tabular$URLs$AMAZON_REVIEWSAMAZON_REVIEWS, sep = ''),
                destfile = paste(filename,'.tgz',sep = ''))

  if(untar)
    untar(paste(filename,'.tgz',sep = ''))

}


#' @title AMAZON_REVIEWS_POLARITY dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download AMAZON_REVIEWS_POLARITY dataset
#' @return None
#' @export
URLs_AMAZON_REVIEWS_POLARITY <- function(filename = 'AMAZON_REVIEWS_POLARITY', untar = TRUE) {

  download.file(paste(tabular$URLs$AMAZON_REVIEWS_POLARITY, sep = ''),
                destfile = paste(filename,'.tgz',sep = ''))

  if(untar)
    untar(paste(filename,'.tgz',sep = ''))

}


#' @title BIWI_HEAD_POSE dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download BIWI_HEAD_POSE dataset
#' @return None
#' @export
URLs_BIWI_HEAD_POSE <- function(filename = 'BIWI_HEAD_POSE', untar = TRUE) {

  download.file(paste(tabular$URLs$BIWI_HEAD_POSE, sep = ''),
                destfile = paste(filename,'.tgz',sep = ''))

  if(untar)
    untar(paste(filename,'.tgz',sep = ''))

}


#' @title CALTECH_101 dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download CALTECH_101 dataset
#' @return None
#' @export
URLs_CALTECH_101 <- function(filename = 'CALTECH_101', untar = TRUE) {

  download.file(paste(tabular$URLs$CALTECH_101, sep = ''),
                destfile = paste(filename,'.tgz',sep = ''))

  if(untar)
    untar(paste(filename,'.tgz',sep = ''))

}


#' @title CAMVID dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download CAMVID dataset
#' @return None
#' @export
URLs_CAMVID <- function(filename = 'CAMVID', untar = TRUE) {

  download.file(paste(tabular$URLs$CAMVID, sep = ''),
                destfile = paste(filename,'.tgz',sep = ''))

  if(untar)
    untar(paste(filename,'.tgz',sep = ''))

}


#' @title CAMVID_TINY dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download CAMVID_TINY dataset
#' @return None
#' @export
URLs_CAMVID_TINY <- function(filename = 'CAMVID_TINY', untar = TRUE) {

  download.file(paste(tabular$URLs$CAMVID_TINY, sep = ''),
                destfile = paste(filename,'.tgz',sep = ''))

  if(untar)
    untar(paste(filename,'.tgz',sep = ''))

}


#' @title CARS dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download CARS dataset
#' @return None
#' @export
URLs_CARS <- function(filename = 'CARS', untar = TRUE) {

  download.file(paste(tabular$URLs$CARS, sep = ''),
                destfile = paste(filename,'.tgz',sep = ''))

  if(untar)
    untar(paste(filename,'.tgz',sep = ''))

}


#' @title CIFAR dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download CIFAR dataset
#' @return None
#' @export
URLs_CIFAR <- function(filename = 'CIFAR', untar = TRUE) {

  download.file(paste(tabular$URLs$CIFAR, sep = ''),
                destfile = paste(filename,'.tgz',sep = ''))

  if(untar)
    untar(paste(filename,'.tgz',sep = ''))

}


#' @title CIFAR_100 dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download CIFAR_100 dataset
#' @return None
#' @export
URLs_CIFAR_100 <- function(filename = 'CIFAR_100', untar = TRUE) {

  download.file(paste(tabular$URLs$CIFAR_100, sep = ''),
                destfile = paste(filename,'.tgz',sep = ''))

  if(untar)
    untar(paste(filename,'.tgz',sep = ''))

}


#' @title COCO_TINY dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download COCO_TINY dataset
#' @return None
#' @export
URLs_COCO_TINY <- function(filename = 'COCO_TINY', untar = TRUE) {

  download.file(paste(tabular$URLs$COCO_TINY, sep = ''),
                destfile = paste(filename,'.tgz',sep = ''))

  if(untar)
    untar(paste(filename,'.tgz',sep = ''))

}


#' @title CUB_200_2011 dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download CUB_200_2011 dataset
#' @return None
#' @export
URLs_CUB_200_2011 <- function(filename = 'CUB_200_2011', untar = TRUE) {

  download.file(paste(tabular$URLs$CUB_200_2011, sep = ''),
                destfile = paste(filename,'.tgz',sep = ''))

  if(untar)
    untar(paste(filename,'.tgz',sep = ''))

}


#' @title DBPEDIA dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download DBPEDIA dataset
#' @return None
#' @export
URLs_DBPEDIA <- function(filename = 'DBPEDIA', untar = TRUE) {

  download.file(paste(tabular$URLs$DBPEDIA, sep = ''),
                destfile = paste(filename,'.tgz',sep = ''))

  if(untar)
    untar(paste(filename,'.tgz',sep = ''))

}


#' @title DOGS dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download DOGS dataset
#' @return None
#' @export
URLs_DOGS <- function(filename = 'DOGS', untar = TRUE) {

  download.file(paste(tabular$URLs$DOGS, sep = ''),
                destfile = paste(filename,'.tgz',sep = ''))

  if(untar)
    untar(paste(filename,'.tgz',sep = ''))

}


#' @title FLOWERS dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download FLOWERS dataset
#' @return None
#' @export
URLs_FLOWERS <- function(filename = 'FLOWERS', untar = TRUE) {

  download.file(paste(tabular$URLs$FLOWERS, sep = ''),
                destfile = paste(filename,'.tgz',sep = ''))

  if(untar)
    untar(paste(filename,'.tgz',sep = ''))

}


#' @title FOOD dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download FOOD dataset
#' @return None
#' @export
URLs_FOOD <- function(filename = 'FOOD', untar = TRUE) {

  download.file(paste(tabular$URLs$FOOD, sep = ''),
                destfile = paste(filename,'.tgz',sep = ''))

  if(untar)
    untar(paste(filename,'.tgz',sep = ''))

}


#' @title HUMAN_NUMBERS dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download HUMAN_NUMBERS dataset
#' @return None
#' @export
URLs_HUMAN_NUMBERS <- function(filename = 'HUMAN_NUMBERS', untar = TRUE) {

  download.file(paste(tabular$URLs$HUMAN_NUMBERS, sep = ''),
                destfile = paste(filename,'.tgz',sep = ''))

  if(untar)
    untar(paste(filename,'.tgz',sep = ''))

}


#' @title IMAGENETTE dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download IMAGENETTE dataset
#' @return None
#' @export
URLs_IMAGENETTE <- function(filename = 'IMAGENETTE', untar = TRUE) {

  download.file(paste(tabular$URLs$IMAGENETTE, sep = ''),
                destfile = paste(filename,'.tgz',sep = ''))

  if(untar)
    untar(paste(filename,'.tgz',sep = ''))

}


#' @title IMAGENETTE_160 dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download IMAGENETTE_160 dataset
#' @return None
#' @export
URLs_IMAGENETTE_160 <- function(filename = 'IMAGENETTE_160', untar = TRUE) {

  download.file(paste(tabular$URLs$IMAGENETTE_160, sep = ''),
                destfile = paste(filename,'.tgz',sep = ''))

  if(untar)
    untar(paste(filename,'.tgz',sep = ''))

}


#' @title IMAGENETTE_320 dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download IMAGENETTE_320 dataset
#' @return None
#' @export
URLs_IMAGENETTE_320 <- function(filename = 'IMAGENETTE_320', untar = TRUE) {

  download.file(paste(tabular$URLs$IMAGENETTE_320, sep = ''),
                destfile = paste(filename,'.tgz',sep = ''))

  if(untar)
    untar(paste(filename,'.tgz',sep = ''))

}


#' @title IMAGEWOOF dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download IMAGEWOOF dataset
#' @return None
#' @export
URLs_IMAGEWOOF <- function(filename = 'IMAGEWOOF', untar = TRUE) {

  download.file(paste(tabular$URLs$IMAGEWOOF, sep = ''),
                destfile = paste(filename,'.tgz',sep = ''))

  if(untar)
    untar(paste(filename,'.tgz',sep = ''))

}


#' @title IMAGEWOOF_160 dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download IMAGEWOOF_160 dataset
#' @return None
#' @export
URLs_IMAGEWOOF_160 <- function(filename = 'IMAGEWOOF_160', untar = TRUE) {

  download.file(paste(tabular$URLs$IMAGEWOOF_160, sep = ''),
                destfile = paste(filename,'.tgz',sep = ''))

  if(untar)
    untar(paste(filename,'.tgz',sep = ''))

}


#' @title IMAGEWOOF_320 dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download IMAGEWOOF_320 dataset
#' @return None
#' @export
URLs_IMAGEWOOF_320 <- function(filename = 'IMAGEWOOF_320', untar = TRUE) {

  download.file(paste(tabular$URLs$IMAGEWOOF_320, sep = ''),
                destfile = paste(filename,'.tgz',sep = ''))

  if(untar)
    untar(paste(filename,'.tgz',sep = ''))

}


#' @title IMDB dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download IMDB dataset
#' @return None
#' @export
URLs_IMDB <- function(filename = 'IMDB', untar = TRUE) {

  download.file(paste(tabular$URLs$IMDB, sep = ''),
                destfile = paste(filename,'.tgz',sep = ''))

  if(untar)
    untar(paste(filename,'.tgz',sep = ''))

}


#' @title IMDB_SAMPLE dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download IMDB_SAMPLE dataset
#' @return None
#' @export
URLs_IMDB_SAMPLE <- function(filename = 'IMDB_SAMPLE', untar = TRUE) {

  download.file(paste(tabular$URLs$IMDB_SAMPLE, sep = ''),
                destfile = paste(filename,'.tgz',sep = ''))

  if(untar)
    untar(paste(filename,'.tgz',sep = ''))

}


#' @title LSUN_BEDROOMS dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download LSUN_BEDROOMS dataset
#' @return None
#' @export
URLs_LSUN_BEDROOMS <- function(filename = 'LSUN_BEDROOMS', untar = TRUE) {

  download.file(paste(tabular$URLs$LSUN_BEDROOMS, sep = ''),
                destfile = paste(filename,'.tgz',sep = ''))

  if(untar)
    untar(paste(filename,'.tgz',sep = ''))

}


#' @title ML_SAMPLE dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download ML_SAMPLE dataset
#' @return None
#' @export
URLs_ML_SAMPLE <- function(filename = 'ML_SAMPLE', untar = TRUE) {

  download.file(paste(tabular$URLs$ML_SAMPLE, sep = ''),
                destfile = paste(filename,'.tgz',sep = ''))

  if(untar)
    untar(paste(filename,'.tgz',sep = ''))

}


#' @title MNIST dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download MNIST dataset
#' @return None
#' @export
URLs_MNIST <- function(filename = 'MNIST', untar = TRUE) {

  download.file(paste(tabular$URLs$MNIST, sep = ''),
                destfile = paste(filename,'.tgz',sep = ''))

  if(untar)
    untar(paste(filename,'.tgz',sep = ''))

}


#' @title MNIST_SAMPLE dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download MNIST_SAMPLE dataset
#' @return None
#' @export
URLs_MNIST_SAMPLE <- function(filename = 'MNIST_SAMPLE', untar = TRUE) {

  download.file(paste(tabular$URLs$MNIST_SAMPLE, sep = ''),
                destfile = paste(filename,'.tgz',sep = ''))

  if(untar)
    untar(paste(filename,'.tgz',sep = ''))

}


#' @title MNIST_TINY dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download MNIST_TINY dataset
#' @return None
#' @export
URLs_MNIST_TINY <- function(filename = 'MNIST_TINY', untar = TRUE) {

  download.file(paste(tabular$URLs$MNIST_TINY, sep = ''),
                destfile = paste(filename,'.tgz',sep = ''))

  if(untar)
    untar(paste(filename,'.tgz',sep = ''))

}


#' @title MNIST_VAR_SIZE_TINY dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download MNIST_VAR_SIZE_TINY dataset
#' @return None
#' @export
URLs_MNIST_VAR_SIZE_TINY <- function(filename = 'MNIST_VAR_SIZE_TINY', untar = TRUE) {

  download.file(paste(tabular$URLs$MNIST_VAR_SIZE_TINY, sep = ''),
                destfile = paste(filename,'.tgz',sep = ''))

  if(untar)
    untar(paste(filename,'.tgz',sep = ''))

}


#' @title MT_ENG_FRA dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download MT_ENG_FRA dataset
#' @return None
#' @export
URLs_MT_ENG_FRA <- function(filename = 'MT_ENG_FRA', untar = TRUE) {

  download.file(paste(tabular$URLs$MT_ENG_FRA, sep = ''),
                destfile = paste(filename,'.tgz',sep = ''))

  if(untar)
    untar(paste(filename,'.tgz',sep = ''))

}


#' @title OPENAI_TRANSFORMER dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download OPENAI_TRANSFORMER dataset
#' @return None
#' @export
URLs_OPENAI_TRANSFORMER <- function(filename = 'OPENAI_TRANSFORMER', untar = TRUE) {

  download.file(paste(tabular$URLs$OPENAI_TRANSFORMER, sep = ''),
                destfile = paste(filename,'.tgz',sep = ''))

  if(untar)
    untar(paste(filename,'.tgz',sep = ''))

}


#' @title PASCAL_2007 dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download PASCAL_2007 dataset
#' @return None
#' @export
URLs_PASCAL_2007 <- function(filename = 'PASCAL_2007', untar = TRUE) {

  download.file(paste(tabular$URLs$PASCAL_2007, sep = ''),
                destfile = paste(filename,'.tgz',sep = ''))

  if(untar)
    untar(paste(filename,'.tgz',sep = ''))

}


#' @title PASCAL_2012 dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download PASCAL_2012 dataset
#' @return None
#' @export
URLs_PASCAL_2012 <- function(filename = 'PASCAL_2012', untar = TRUE) {

  download.file(paste(tabular$URLs$PASCAL_2012, sep = ''),
                destfile = paste(filename,'.tgz',sep = ''))

  if(untar)
    untar(paste(filename,'.tgz',sep = ''))

}


#' @title PETS dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download PETS dataset
#' @return None
#' @export
URLs_PETS <- function(filename = 'PETS', untar = TRUE) {

  download.file(paste(tabular$URLs$PETS, sep = ''),
                destfile = paste(filename,'.tgz',sep = ''))

  if(untar)
    untar(paste(filename,'.tgz',sep = ''))

}


#' @title PLANET_SAMPLE dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download PLANET_SAMPLE dataset
#' @return None
#' @export
URLs_PLANET_SAMPLE <- function(filename = 'PLANET_SAMPLE', untar = TRUE) {

  download.file(paste(tabular$URLs$PLANET_SAMPLE, sep = ''),
                destfile = paste(filename,'.tgz',sep = ''))

  if(untar)
    untar(paste(filename,'.tgz',sep = ''))

}


#' @title PLANET_TINY dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download PLANET_TINY dataset
#' @return None
#' @export
URLs_PLANET_TINY <- function(filename = 'PLANET_TINY', untar = TRUE) {

  download.file(paste(tabular$URLs$PLANET_TINY, sep = ''),
                destfile = paste(filename,'.tgz',sep = ''))

  if(untar)
    untar(paste(filename,'.tgz',sep = ''))

}


#' @title S3_COCO dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download S3_COCO dataset
#' @return None
#' @export
URLs_S3_COCO <- function(filename = 'S3_COCO', untar = TRUE) {

  download.file(paste(tabular$URLs$S3_COCO, sep = ''),
                destfile = paste(filename,'.tgz',sep = ''))

  if(untar)
    untar(paste(filename,'.tgz',sep = ''))

}


#' @title S3_IMAGE dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download S3_IMAGE dataset
#' @return None
#' @export
URLs_S3_IMAGE <- function(filename = 'S3_IMAGE', untar = TRUE) {

  download.file(paste(tabular$URLs$S3_IMAGE, sep = ''),
                destfile = paste(filename,'.tgz',sep = ''))

  if(untar)
    untar(paste(filename,'.tgz',sep = ''))

}


#' @title S3_IMAGELOC dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download S3_IMAGELOC dataset
#' @return None
#' @export
URLs_S3_IMAGELOC <- function(filename = 'S3_IMAGELOC', untar = TRUE) {

  download.file(paste(tabular$URLs$S3_IMAGELOC, sep = ''),
                destfile = paste(filename,'.tgz',sep = ''))

  if(untar)
    untar(paste(filename,'.tgz',sep = ''))

}


#' @title S3_MODEL dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download S3_MODEL dataset
#' @return None
#' @export
URLs_S3_MODEL <- function(filename = 'S3_MODEL', untar = TRUE) {

  download.file(paste(tabular$URLs$S3_MODEL, sep = ''),
                destfile = paste(filename,'.tgz',sep = ''))

  if(untar)
    untar(paste(filename,'.tgz',sep = ''))

}


#' @title S3_NLP dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download S3_NLP dataset
#' @return None
#' @export
URLs_S3_NLP <- function(filename = 'S3_NLP', untar = TRUE) {

  download.file(paste(tabular$URLs$S3_NLP, sep = ''),
                destfile = paste(filename,'.tgz',sep = ''))

  if(untar)
    untar(paste(filename,'.tgz',sep = ''))

}


#' @title SKIN_LESION dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download SKIN_LESION dataset
#' @return None
#' @export
URLs_SKIN_LESION <- function(filename = 'SKIN_LESION', untar = TRUE) {

  download.file(paste(tabular$URLs$SKIN_LESION, sep = ''),
                destfile = paste(filename,'.tgz',sep = ''))

  if(untar)
    untar(paste(filename,'.tgz',sep = ''))

}


#' @title SOGOU_NEWS dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download SOGOU_NEWS dataset
#' @return None
#' @export
URLs_SOGOU_NEWS <- function(filename = 'SOGOU_NEWS', untar = TRUE) {

  download.file(paste(tabular$URLs$SOGOU_NEWS, sep = ''),
                destfile = paste(filename,'.tgz',sep = ''))

  if(untar)
    untar(paste(filename,'.tgz',sep = ''))

}


#' @title WIKITEXT dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download WIKITEXT dataset
#' @return None
#' @export
URLs_WIKITEXT <- function(filename = 'WIKITEXT', untar = TRUE) {

  download.file(paste(tabular$URLs$WIKITEXT, sep = ''),
                destfile = paste(filename,'.tgz',sep = ''))

  if(untar)
    untar(paste(filename,'.tgz',sep = ''))

}


#' @title WIKITEXT_TINY dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download WIKITEXT_TINY dataset
#' @return None
#' @export
URLs_WIKITEXT_TINY <- function(filename = 'WIKITEXT_TINY', untar = TRUE) {

  download.file(paste(tabular$URLs$WIKITEXT_TINY, sep = ''),
                destfile = paste(filename,'.tgz',sep = ''))

  if(untar)
    untar(paste(filename,'.tgz',sep = ''))

}


#' @title WT103_BWD dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download WT103_BWD dataset
#' @return None
#' @export
URLs_WT103_BWD <- function(filename = 'WT103_BWD', untar = TRUE) {

  download.file(paste(tabular$URLs$WT103_BWD, sep = ''),
                destfile = paste(filename,'.tgz',sep = ''))

  if(untar)
    untar(paste(filename,'.tgz',sep = ''))

}


#' @title WT103_FWD dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download WT103_FWD dataset
#' @return None
#' @export
URLs_WT103_FWD <- function(filename = 'WT103_FWD', untar = TRUE) {

  download.file(paste(tabular$URLs$WT103_FWD, sep = ''),
                destfile = paste(filename,'.tgz',sep = ''))

  if(untar)
    untar(paste(filename,'.tgz',sep = ''))

}


#' @title YAHOO_ANSWERS dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download YAHOO_ANSWERS dataset
#' @return None
#' @export
URLs_YAHOO_ANSWERS <- function(filename = 'YAHOO_ANSWERS', untar = TRUE) {

  download.file(paste(tabular$URLs$YAHOO_ANSWERS, sep = ''),
                destfile = paste(filename,'.tgz',sep = ''))

  if(untar)
    untar(paste(filename,'.tgz',sep = ''))

}


#' @title YELP_REVIEWS dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download YELP_REVIEWS dataset
#' @return None
#' @export
URLs_YELP_REVIEWS <- function(filename = 'YELP_REVIEWS', untar = TRUE) {

  download.file(paste(tabular$URLs$YELP_REVIEWS, sep = ''),
                destfile = paste(filename,'.tgz',sep = ''))

  if(untar)
    untar(paste(filename,'.tgz',sep = ''))

}


#' @title YELP_REVIEWS_POLARITY dataset
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download YELP_REVIEWS_POLARITY dataset
#' @return None
#' @export
URLs_YELP_REVIEWS_POLARITY <- function(filename = 'YELP_REVIEWS_POLARITY', untar = TRUE) {

  download.file(paste(tabular$URLs$YELP_REVIEWS_POLARITY, sep = ''),
                destfile = paste(filename,'.tgz',sep = ''))

  if(untar)
    untar(paste(filename,'.tgz',sep = ''))

}

#' @title MOVIE_LENS_ML_100k dataset
#'
#' @param filename the name of the file
#' @param unzip logical, whether to unzip the '.zip' file
#'
#' @description download MOVIE_LENS_ML_100k dataset
#' @return None
#' @export
URLs_MOVIE_LENS_ML_100k <- function(filename = 'ml-100k', unzip = TRUE) {

  download.file('http://files.grouplens.org/datasets/movielens/ml-100k.zip',
                destfile = paste(filename,'.zip',sep = ''))

  if(unzip)
    unzip(paste(filename,'.zip',sep = ''))

}

#' @title SIIM_SMALL
#'
#' @param filename the name of the file
#' @param untar logical, whether to untar the '.tgz' file
#'
#' @description download YELP_REVIEWS_POLARITY dataset
#' @return None
#' @export
URLs_SIIM_SMALL <- function(filename = 'SIIM_SMALL', untar = TRUE) {

  download.file(paste(tabular$URLs$SIIM_SMALL, sep = ''),
                destfile = paste(filename,'.tgz',sep = ''))

  if(untar)
    untar(paste(filename,'.tgz',sep = ''))

}
