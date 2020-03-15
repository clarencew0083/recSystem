#' Test out running Python
#'
#' @title  recommend
#' @description Recommend a meta model for a classification dataset
#' @param sampsize The size of the sample
#' @param nsim     The number of simulations
#'
#' @return A seaborn kdeplot (kernel density estimate plot)
#' @export
#' @import reticulate
recommend <- function(meta= meta_features, metric= recall) {
  
  df <- readr::read_csv(file.choose())
  #in_dir <- dirname(df)
  #df = readr::read_csv(file.choose())
  #print(typeof(df))
  response <- readline(prompt="Enter name of response column: ")
  out <- py$rec_sys(df, response, meta_features, recall)
  paste("Reccomended algoritm: ", out[[2]])
  return(out)
}

#' Test out running Python
#'
#' @title  recommend
#' @description Recommend a meta model for a classification dataset
#' @param sampsize The size of the sample
#' @param nsim     The number of simulations
#'
#' @return A seaborn kdeplot (kernel density estimate plot)
#' @import reticulate
recommend_alg <- function(df, response, meta= meta_features, metric= recall) {
  
  #df <- readr::read_csv(file.choose())
  #in_dir <- dirname(df)
  #df = readr::read_csv(file.choose())
  #print(typeof(df))
  #response <- readline(prompt="Enter name of response column: ")
  lst <- py$rec_sys(df, response, meta_features, recall)
  return(lst[[2]])
}

#' Test out running Python
#'
#' @title  recommend
#' @description Recommend a meta model for a classification dataset
#' @param sampsize The size of the sample
#' @param nsim     The number of simulations
#'
#' @return A seaborn kdeplot (kernel density estimate plot)
#' @import reticulate
recommend_shiny <- function(df, response, meta= meta_features, metric= recall) {
  
  #df <- readr::read_csv(file.choose())
  #in_dir <- dirname(df)
  #df = readr::read_csv(file.choose())
  #print(typeof(df))
  #response <- readline(prompt="Enter name of response column: ")
  lst <- py$rec_sys(df, response, meta_features, recall)
  return(lst[[1]])
  
}



install_sklearn <- function(method = "auto", conda = "auto") {
  reticulate::py_install("scikit-learn", method = method, conda = conda)
}