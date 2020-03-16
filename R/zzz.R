.onLoad <- function(libname, pkgname){
  # use superassignment to update global reference to scipy
  #create_models <<- reticulate::import_from_path(module = "create_models_bin_response", 
                                                 #path = system.file("python", package = "recSystem"))
  constants <<- reticulate::import_from_path(module = "_01_constants", 
                                                 path = system.file("python", package = "recSystem"))
  
  mf <<- reticulate::import_from_path(module = "_02_my_functions", 
                                                 path = system.file("python", package = "recSystem"))
  
  prep <<- reticulate::import_from_path(module = "_03_prepare_data", 
                                                 path = system.file("python", package = "recSystem"))
  
  algs <<- reticulate::import_from_path(module = "_04_algorithms", 
                                                 path = system.file("python", package = "recSystem"))
  
  
  have_scipy <- py_module_available("sklearn")
  if (!have_scipy)
    print("sklearn not available. Install using some function")
  #else
    #print("hi")
  
  sklearn <<- reticulate::import("sklearn", delay_load = TRUE)
  numpy <<- reticulate::import("numpy", delay_load = TRUE)
    
  py_file = system.file("python", "python.py", package = "recSystem")
    
  reticulate::source_python(py_file)

}


install_sklearn <- function(method = "auto", conda = "auto") {
  reticulate::py_install("scikit-learn", method = method, conda = conda)
}
#.onUnload <- function(libpath) {
#  library.dynam.unload("recSystem", libpath)
#}