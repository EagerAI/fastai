
#' @title emb_sz_rule
#'
#' @description Rule of thumb to pick embedding size corresponding to `n_cat`
#'
#' @details
#'
#' @param n_cat n_cat
#'
#' @export
emb_sz_rule <- function(n_cat) {

  tabular$emb_sz_rule(
    n_cat = n_cat
  )

}


#' @title get_emb_sz
#'
#' @description Get default embedding size from `TabularPreprocessor` `proc` or the ones in `sz_dict`
#'
#' @details
#'
#' @param to to
#' @param sz_dict sz_dict
#'
#' @export
get_emb_sz <- function(to, sz_dict = NULL) {

  tabular$get_emb_sz(
    to = to,
    sz_dict = sz_dict
  )

}


#' @title tabular_config
#'
#' @description Convenience function to easily create a config for `TabularModel`
#'
#' @details
#'
#' @param ps ps
#' @param embed_p embed_p
#' @param y_range y_range
#' @param use_bn use_bn
#' @param bn_final bn_final
#' @param bn_cont bn_cont
#' @param act_cls act_cls
#'
#' @export
tabular_config <- function(ps = NULL, embed_p = 0.0, y_range = NULL,
                           use_bn = TRUE, bn_final = FALSE,
                           bn_cont = TRUE, act_cls = nn$ReLU(inplace = TRUE)) {

  tabular$tabular_config(
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
#' @details
#'
#' @param emb_szs emb_szs
#' @param n_cont n_cont
#' @param out_sz out_sz
#' @param layers layers
#' @param ps ps
#' @param embed_p embed_p
#' @param y_range y_range
#' @param use_bn use_bn
#' @param bn_final bn_final
#' @param bn_cont bn_cont
#' @param act_cls act_cls
#'
#' @export
TabularModel <- function(emb_szs, n_cont, out_sz, layers, ps = NULL,
                         embed_p = 0.0, y_range = NULL, use_bn = TRUE, bn_final = FALSE,
                         bn_cont = TRUE, act_cls = nn$ReLU(inplace = TRUE)) {

  if(missing(emb_szs) & missing(n_cont) & missing(out_sz) & layers) {
    invisible(tabular$TabularModel)
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

    do.call(tabular$TabularModel, args)
  }

}


