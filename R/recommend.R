#'
#' @title  recommend
#' @description Recommend a meta model for a classification dataset. Dataset is selected using dialog
#' @param meta_features The extracted meta features of various datasets 
#' @param recall     The known recall of the datasets in the meta_features dataframe
#'
#' @return A collection that contains the cleaned dataframe and the recommended algorithm as a string
#' @export
#' @import reticulate
#' @examples 
#' out<- recommend()
#' View(out[[1]])
#' print(out[[2]])
recommend <- function(meta= meta_features, metric= recall) {
  
  df <- readr::read_csv(file.choose())
  response <- readline(prompt="Enter name of response column: ")
  out <- py$rec_sys(df, response, meta_features, recall)
  paste("Reccomended algoritm: ", out[[2]])
  return(out)
}

#'
#' @title  recommend2
#' @description Recommend a meta model for a classification dataset
#' @param  df Dataset in tabular form
#' @param response The target column of the dataset
#' @param meta_features The extracted meta features of various datasets 
#' @param recall     The known recall of the datasets in the meta_features dataframe
#'
#' @return A collection that contains the cleaned dataframe and the recommended algorithm as a string
#' @export
#' @import reticulate
#' @examples 
#' out<- recommend2(math_placement, "CourseSuccess")
#' View(out[[1]])
#' print(out[[2]])
recommend2 <- function(df, response, meta= meta_features, metric= recall) {
  
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
  
  lst <- py$rec_sys(df, response, meta_features, recall)
  return(lst[[1]])
  
}

#'
#' @title  install_sklearn
#' @description Installs scikit-learn into reticluate python enviornment
#' @return The cleaned dataset
#' @import reticulate
install_sklearn <- function(method = "auto", conda = "auto") {
  reticulate::py_install("scikit-learn", method = method, conda = conda)
}