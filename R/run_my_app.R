#' @export
#' @importFrom  utils View
#' @importFrom  shiny shinyAppDir
run_my_app  <-
  function(app_name = NULL, 
           theme = 'flatly',
           width = '100%',
           height = '800px',
           more_opts = list(NA),
           launch.browser = TRUE,...)
  {
    
    valid_apps <- list.files(system.file("apps", package = "recSystem"))
    
    valid_apps_df <- data.frame(valid_apps)
    colnames(valid_apps_df) <- 'Valid apps'
    
    if (missing(app_name) || !nzchar(app_name) || !app_name %in% valid_apps) {
      
      stop(paste0('Please run `run_my_app()` with a valid app as an argument.\n',
                  "See table for Valid apps"),
           utils::View(valid_apps_df),
           call. = FALSE)
  }
    
    
    dir <- system.file('apps', app_name, package = 'recSystem')

    
    shiny::shinyAppDir(appDir = dir, 
                       options = list(height = height, 
                                      width = width,
                                      launch.browser = launch.browser,...))
    
  }