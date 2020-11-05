
#' @title Emb_sz_rule
#'
#' @description Rule of thumb to pick embedding size corresponding to `n_cat`
#'
#'
#' @param n_cat n_cat
#' @return None
#' @export
emb_sz_rule <- function(n_cat) {

  tabular()$emb_sz_rule(
    n_cat = n_cat
  )

}


#' @title Get_emb_sz
#'
#' @description Get default embedding size from `TabularPreprocessor` `proc` or the ones in `sz_dict`
#'
#'
#' @param to to
#' @param sz_dict dictionary size
#' @return None
#' @export
get_emb_sz <- function(to, sz_dict = NULL) {

  tabular()$get_emb_sz(
    to = to,
    sz_dict = sz_dict
  )

}


#' @title Tabular_config
#'
#' @description Convenience function to easily create a config for `TabularModel`
#'
#'
#' @param ps ps
#' @param embed_p embed proportion
#' @param y_range y_range
#' @param use_bn use batch normalization
#' @param bn_final batch normalization final
#' @param bn_cont batch normalization
#' @param act_cls activation
#' @return None
#' @export
tabular_config <- function(ps = NULL, embed_p = 0.0, y_range = NULL,
                           use_bn = TRUE, bn_final = FALSE,
                           bn_cont = TRUE, act_cls = nn$ReLU(inplace = TRUE)) {

  tabular()$tabular_config(
    ps = ps,
    embed_p = embed_p,
    y_range = y_range,
    use_bn = use_bn,
    bn_final = bn_final,
    bn_cont = bn_cont,
    act_cls = act_cls
  )

}


#' @title TabularModel
#'
#' @description Basic model for tabular data.
#'
#'
#' @param emb_szs embedding size
#' @param n_cont number of cont
#' @param out_sz output size
#' @param layers layers
#' @param ps ps
#' @param embed_p embed proportion
#' @param y_range y range
#' @param use_bn use batch normalization
#' @param bn_final batch normalization final
#' @param bn_cont batch normalization cont
#' @param act_cls activation
#' @return None
#' @export
TabularModel <- function(emb_szs, n_cont, out_sz, layers, ps = NULL,
                         embed_p = 0.0, y_range = NULL, use_bn = TRUE, bn_final = FALSE,
                         bn_cont = TRUE, act_cls = nn$ReLU(inplace = TRUE)) {

  if(missing(emb_szs) & missing(n_cont) & missing(out_sz) & layers) {
    invisible(tabular()$TabularModel)
  } else {
    args <- list(
      emb_szs = emb_szs,
      n_cont = n_cont,
      out_sz = out_sz,
      layers = layers,
      ps = ps,
      embed_p = embed_p,
      y_range = y_range,
      use_bn = use_bn,
      bn_final = bn_final,
      bn_cont = bn_cont,
      act_cls = act_cls
    )

    do.call(tabular()$TabularModel, args)
  }

}


