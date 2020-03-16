#' @title  Meta Features data set
#' @name meta_features
#' @description A dataset containing the metafeatures of 15 datasets. These datasets are used to construct the meta model
#'
#' @format A dataset with 15 rows and 13 columns
#' 
#' \describe{
#'    \item{Name}{The name of a particular dataset}
#'    \item{Rows}{}
#'    \item{Columns}{}
#'    \item{Rows-Cols Ratio}{}
#'    \item{Number Discrete}{}
#'    \item{Max num factors}{}
#'    \item{Min num factors}{}
#'    \item{Avg num factors}{}
#'    \item{Number Continuous}{}
#'    \item{Gradient-Avg}{}
#'    \item{Gradient-Min}{}
#'    \item{Gradient-Max}{}
#'    \item{Gradient-Std}{}
#'    }
NULL


#' @title  Recall data set
#' @name recall
#' @description 
#' #' A dataset containing the metafeatures snd recall of the algorithms SVM, KNN, and naiive bayes classifier of 15 datasets. These datasets are used to construct the meta model
#' 
#' @format A \code{tibble} with 15 rows and 16 columns
#' 
#' \describe{
#'    \item{Name}{The name of a particular dataset}
#'    \item{Rows}{}
#'    \item{Columns}{}
#'    \item{Rows-Cols Ratio}{}
#'    \item{Number Discrete}{}
#'    \item{Max num factors}{}
#'    \item{Min num factors}{}
#'    \item{Avg num factors}{}
#'    \item{Number Continuous}{}
#'    \item{Gradient-Avg}{}
#'    \item{Gradient-Min}{}
#'    \item{Gradient-Max}{}
#'    \item{Gradient-Std}{}
#'    \item{SVM}{Recall when using Support Vector Machine classifier on dataset}
#'    \item{KNN}{Recall when using K Nearest Neighbors classifier on dataset}
#'    \item{NB}{Recall when using Naiive Bayes classifier on dataset}
#'    }
NULL

#' @title  Math Placement Exam Results
#' @name math_placement
#' @description 
#' Response is courseSucess
#' 
#' @format A dataset with 2696 rows and 15 columns
#' 
#' \describe{
#'    \item{Student}{Identification number for each student}
#'    \item{Gender}{0=Female, 1=Male}
#'    \item{PSATM}{PSAT score in Math}
#'    \item{SATM}{SAT score in Math}
#'    \item{ACTM}{ACTM score in Math}
#'    \item{Rank}{Adjusted rank in HS class}
#'    \item{Size}{Number of students in HS class}
#'    \item{GPAadj}{Adjusted GPA}
#'    \item{PlcmtScore}{Score on math placement exam}
#'    \item{Recommends}{Recommended course: R0 R01 R1 R12 R2 R3 R4 R6 R8}
#'    \item{Grade}{Course grade}
#'    \item{Rectaken}{1=recommended course, 0=otherwise}
#'    \item{TooHigh}{1=took course above recommended, 0=otherwise}
#'    \item{TooLow}{1=took course above recommended, 0=otherwise}
#'    \item{CourseSuccess}{1=B or better grade, 0=grade below B}
#'    }
#
#' @source \url{http://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/MathPlacement.html}
NULL


#' @title  Urine data set
#' @name urine
#' @description 
#' A very large dataset
#' 
#' @format A \code{tibble} with 79 rows and 7 columns
#' 
#' \describe{
#'    \item{r}{Indicator of the presence of calcium oxalate crystals.}
#'    \item{gravity}{The specific gravity of the urine}
#'    \item{ph}{The pH reading of the urine.}
#'    \item{osmo}{The osmolarity of the urine. Osmolarity is proportional to the concentration of molecules in solution}
#'    \item{cond}{The conductivity of the urine. Conductivity is proportional to the concentration of charged ions in solution.}
#'    \item{urea}{The urea concentration in millimoles per litre.}
#'    \item{calc}{The calcium concentration in millimoles per litre.}
#'    }
#'    
#' @source \url{http://vincentarelbundock.github.io/Rdatasets/doc/boot/urine.html}
NULL

