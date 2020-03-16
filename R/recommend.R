#' Test out running Python
#'
#' @title  recommend
#' @description Recommend a meta model for a classification dataset
#' @param meta_features The extracted meta features of various datasets 
#' @param recall     The known recall of the datasets in the meta_features dataframe
#'
#' @return A collection that contains the cleaned dataframe and the reccomended algorithm as a string
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

#'
#' @title  recommend_alg
#' @description Recommend a meta model for a classification dataset. Used internally by R shiny app
#' @param df The dataframe loaded  by the user 
#' @param response    The target column of thedataframe loaded  by the user
#' @param meta_features The preextracted meta features of various datasets 
#' @param recall     The known recall of the datasets in the meta_features dataframe
#'
#' @return The reccomended algorithm as a string
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
#' @title  recommend_shiny
#' @description Recommend a meta model for a classification dataset. Used internally by R shiny app
#' @param df The dataframe loaded  by the user 
#' @param response    The target column of thedataframe loaded  by the user
#' @param meta_features The preextracted meta features of various datasets 
#' @param recall     The known recall of the datasets in the meta_features dataframe
#'
#' @return The cleaned dataset
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

#' Test out running Python
#'
#' @title  install_sklearn
#' @description Installs scikit-learn into reticluate python enviornment
#' @return The cleaned dataset
#' @import reticulate
install_sklearn <- function(method = "auto", conda = "auto") {
  reticulate::py_install("scikit-learn", method = method, conda = conda)
}